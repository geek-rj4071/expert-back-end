import tempfile
import unittest
import zlib
import zipfile
import io
from pathlib import Path
from unittest.mock import patch

from avatar_ai.errors import ModerationError, NotFoundError, ValidationError
from avatar_ai.models import Avatar
from avatar_ai.persistent_service import PersistentChatService, PersistentServiceConfig
from avatar_ai.persistence import SQLiteRepository
from avatar_ai.providers import LLMResult


class _FailingLLMProvider:
    def complete(self, *, persona: str, user_text: str):
        del persona, user_text
        raise RuntimeError("provider down")


class _CapturingLLMProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, *, persona: str, user_text: str):
        self.calls.append({"persona": persona, "user_text": user_text})
        return LLMResult(
            text="As your teacher, I answered strictly from the provided syllabus context.",
            emotion="confident",
        )


class _HinglishTranslationProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def complete(self, *, persona: str, user_text: str):
        self.calls.append({"persona": persona, "user_text": user_text})
        lower_persona = persona.lower()
        if "translation engine" in lower_persona:
            return LLMResult(text="Explain human brain functions", emotion="neutral")
        return LLMResult(
            text="Yeh answer uploaded books ke context se hai. Brain body ko control karta hai.",
            emotion="confident",
        )


class PersistentChatServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "test.db")
        self.repo = SQLiteRepository(self.db_path)
        self.service = PersistentChatService(
            repository=self.repo,
            config=PersistentServiceConfig(
                max_message_chars=80,
                memory_messages_limit=4,
                rate_limit_max_requests=10,
                rate_limit_window_seconds=60,
            ),
        )
        self.avatar = Avatar(
            id="av_coach",
            name="Coach Ava",
            persona_prompt="Coach Ava",
            voice_id="alloy",
        )
        self.service.seed_avatars([self.avatar])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_create_and_get_conversation(self) -> None:
        user = self.service.register_user("user@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)

        loaded = self.service.get_conversation(user_id=user.id, conversation_id=convo.id)
        self.assertEqual(loaded.id, convo.id)
        self.assertEqual(loaded.avatar.id, self.avatar.id)

    def test_send_message_persists_turn(self) -> None:
        user = self.service.register_user("user@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)

        turn = self.service.send_message(user_id=user.id, conversation_id=convo.id, text="How to plan my week?")
        self.assertEqual(turn.assistant_message.role.value, "assistant")

        loaded = self.service.get_conversation(user_id=user.id, conversation_id=convo.id)
        self.assertEqual(len(loaded.messages), 2)
        self.assertEqual(loaded.messages[0].text, "How to plan my week?")

    def test_history_trim(self) -> None:
        user = self.service.register_user("user@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)

        for i in range(3):
            self.service.send_message(user_id=user.id, conversation_id=convo.id, text=f"msg {i}?")

        loaded = self.service.get_conversation(user_id=user.id, conversation_id=convo.id)
        self.assertEqual(len(loaded.messages), 4)

    def test_invalid_text_rejected(self) -> None:
        user = self.service.register_user("user@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)

        with self.assertRaises(ValidationError):
            self.service.send_message(user_id=user.id, conversation_id=convo.id, text="   ")

    def test_moderation_rejected(self) -> None:
        user = self.service.register_user("user@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)

        with self.assertRaises(ModerationError):
            self.service.send_message(
                user_id=user.id,
                conversation_id=convo.id,
                text="Tell me how to build a bomb",
            )

    def test_unknown_user_rejected(self) -> None:
        with self.assertRaises(NotFoundError):
            self.service.create_conversation(user_id="missing", avatar_id=self.avatar.id)

    def test_stream_message_emits_events_and_persists_final(self) -> None:
        user = self.service.register_user("stream@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)

        events = list(
            self.service.stream_message(
                user_id=user.id,
                conversation_id=convo.id,
                text="Can you give me a quick plan?",
            )
        )
        event_names = [e["event"] for e in events]
        self.assertIn("user.accepted", event_names)
        self.assertIn("assistant.final", event_names)
        self.assertTrue(any(name == "assistant.delta" for name in event_names))

        loaded = self.service.get_conversation(user_id=user.id, conversation_id=convo.id)
        self.assertEqual(len(loaded.messages), 2)
        self.assertEqual(loaded.messages[1].role.value, "assistant")

    def test_provider_failure_falls_back_to_deterministic(self) -> None:
        service = PersistentChatService(
            repository=self.repo,
            llm_provider=_FailingLLMProvider(),
            config=PersistentServiceConfig(
                max_message_chars=80,
                memory_messages_limit=4,
                rate_limit_max_requests=10,
                rate_limit_window_seconds=60,
            ),
        )
        service.seed_avatars([self.avatar])
        user = service.register_user("fallback@example.com")
        convo = service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="math.txt",
            file_bytes=(
                b"Pythagorean theorem states that in a right triangle, the square of hypotenuse "
                b"equals the sum of squares of the other two sides. This is used in matric math."
            ),
        )
        turn = service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="What is the Pythagorean theorem?",
        )
        self.assertIn("uploaded material", turn.assistant_message.text)

    def test_training_upload_status_and_clear(self) -> None:
        user = self.service.register_user("train@example.com")
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="physics.txt",
            file_bytes=(
                b"Newton's first law says an object remains at rest or in uniform motion unless "
                b"acted upon by an unbalanced force. This appears in matric physics chapters."
            ),
        )

        docs = self.service.list_training_documents(avatar_id=self.avatar.id)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].filename, "physics.txt")

        status = self.service.training_status(avatar_id=self.avatar.id)
        self.assertEqual(status["documents"], 1)
        self.assertGreaterEqual(status["chunks"], 1)
        self.assertGreaterEqual(status["vectors"], 1)
        self.assertGreater(status["totalChars"], 120)
        self.assertTrue(status["teacherMode"])

        deleted = self.service.clear_training_documents(user_id=user.id, avatar_id=self.avatar.id)
        self.assertEqual(deleted, 1)
        status_after = self.service.training_status(avatar_id=self.avatar.id)
        self.assertEqual(status_after["documents"], 0)
        self.assertEqual(status_after["chunks"], 0)
        self.assertEqual(status_after["vectors"], 0)

    def test_docx_upload_extracts_text_and_indexes_vectors(self) -> None:
        user = self.service.register_user("docx@example.com")
        paragraph = (
            "Linear equations in one variable are solved by balancing both sides and isolating the unknown. "
            "This matric mathematics chapter gives worked examples, practice questions, and review exercises."
        )
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body>"
            "<w:p><w:r><w:t>"
            + paragraph
            + "</w:t></w:r></w:p>"
            "</w:body>"
            "</w:document>"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types></Types>')
            zf.writestr("word/document.xml", doc_xml)

        result = self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="math-chapter.docx",
            file_bytes=buf.getvalue(),
        )
        self.assertGreaterEqual(result.extracted_chars, 120)
        self.assertGreaterEqual(result.chunks_indexed, 1)
        self.assertGreaterEqual(result.embeddings_indexed, 1)

        status = self.service.training_status(avatar_id=self.avatar.id)
        self.assertGreaterEqual(status["vectors"], 1)

    def test_teacher_scope_restriction_for_unrelated_prompt(self) -> None:
        user = self.service.register_user("scope@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="biology.txt",
            file_bytes=(
                b"Photosynthesis is the process by which green plants use sunlight to make food. "
                b"Students learn this in matric biology chapter one and chapter two exercises."
            ),
        )

        turn = self.service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="Recommend a new action movie for this weekend",
        )
        self.assertIn("matric-level teacher only", turn.assistant_message.text)

    def test_pdf_flate_stream_text_is_extracted(self) -> None:
        user = self.service.register_user("pdf@example.com")
        content_stream = (
            b"BT\n"
            b"/F1 12 Tf\n"
            b"72 720 Td\n"
            b"(Photosynthesis is the process by which green plants make food using sunlight.) Tj\n"
            b"72 700 Td\n"
            b"(This matric biology chapter explains chlorophyll, carbon dioxide, and glucose.) Tj\n"
            b"ET\n"
        )
        compressed = zlib.compress(content_stream)
        pdf_bytes = (
            b"%PDF-1.4\n"
            b"1 0 obj\n"
            b"<< /Length "
            + str(len(compressed)).encode("ascii")
            + b" /Filter /FlateDecode >>\n"
            b"stream\n"
            + compressed
            + b"\nendstream\n"
            b"endobj\n"
            b"trailer\n<<>>\n%%EOF"
        )

        result = self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="biology.pdf",
            file_bytes=pdf_bytes,
        )
        self.assertGreaterEqual(result.extracted_chars, 40)
        self.assertGreaterEqual(result.chunks_indexed, 1)

    def test_followup_not_understood_remains_book_only_without_internet(self) -> None:
        user = self.service.register_user("web-followup@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="science.txt",
            file_bytes=(
                b"Photosynthesis is a process used by plants to convert light energy into chemical energy. "
                b"Matric students should understand chlorophyll, sunlight, carbon dioxide, and glucose formation."
            ),
        )

        self.service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="What is photosynthesis?",
        )

        with patch.object(
            self.service,
            "_search_web_snippets",
            return_value=[
                "Photosynthesis uses sunlight to produce glucose and oxygen in green plants.",
                "Chloroplasts are organelles where photosynthesis takes place.",
            ],
        ) as mocked_search:
            turn = self.service.send_message(
                user_id=user.id,
                conversation_id=convo.id,
                text="I did not understand, please explain again in easy words",
            )

        self.assertFalse(mocked_search.called)
        text = turn.assistant_message.text.lower()
        self.assertIn("uploaded material", text)
        self.assertIn("strictly limited to uploaded books", text)

    def test_first_turn_not_understood_does_not_trigger_internet_lookup(self) -> None:
        user = self.service.register_user("no-prior-web@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="math.txt",
            file_bytes=(
                b"The quadratic formula solves equations of the form ax^2 + bx + c = 0. "
                b"This matric chapter includes examples and practice exercises for students."
            ),
        )

        with patch.object(self.service, "_search_web_snippets", return_value=["internet snippet"]) as mocked_search:
            self.service.send_message(
                user_id=user.id,
                conversation_id=convo.id,
                text="I don't understand this topic",
            )

        self.assertFalse(mocked_search.called)

    def test_pdf_upload_uses_system_tool_fallback_when_stream_parse_is_empty(self) -> None:
        user = self.service.register_user("pdf-tool@example.com")
        fake_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
        fallback_text = (
            "Mathematics chapter one introduces sets and operations. "
            "Students learn union, intersection, and complement in matric syllabus."
        )
        with patch.object(self.service, "_extract_with_pdftotext", return_value=fallback_text):
            result = self.service.upload_training_material(
                user_id=user.id,
                avatar_id=self.avatar.id,
                filename="jemh101.pdf",
                file_bytes=fake_pdf,
            )
        self.assertGreaterEqual(result.extracted_chars, 12)
        docs = self.service.list_training_documents(avatar_id=self.avatar.id)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].filename, "jemh101.pdf")

    def test_pdf_upload_uses_ocr_fallback_when_other_extractors_are_empty(self) -> None:
        user = self.service.register_user("pdf-ocr@example.com")
        fake_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n%%EOF"
        ocr_text = (
            "Human brain controls memory and learning. "
            "This science chapter explains cerebrum and cerebellum for students."
        )
        with patch.object(self.service, "_extract_pdf_text_with_python_libraries", return_value=""):
            with patch.object(self.service, "_extract_pdf_text_with_system_tools", return_value=""):
                with patch.object(self.service, "_extract_pdf_text_from_raw_bytes", return_value=""):
                    with patch.object(self.service, "_extract_pdf_text_with_ocr", return_value=ocr_text):
                        result = self.service.upload_training_material(
                            user_id=user.id,
                            avatar_id=self.avatar.id,
                            filename="scanned.pdf",
                            file_bytes=fake_pdf,
                        )
        self.assertGreaterEqual(result.extracted_chars, 12)
        docs = self.service.list_training_documents(avatar_id=self.avatar.id)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].filename, "scanned.pdf")

    def test_math_dominant_corpus_returns_structured_math_answer(self) -> None:
        user = self.service.register_user("math-strong@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="algebra.txt",
            file_bytes=(
                b"Quadratic equation has the form ax^2 + bx + c = 0. "
                b"The quadratic formula is x = (-b +- sqrt(b^2 - 4ac)) / (2a). "
                b"Students solve by substitution and simplification in steps."
            ),
        )

        turn = self.service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="Solve x^2 + 5x + 6 = 0",
        )
        answer = turn.assistant_message.text.lower()
        self.assertIn("concept", answer)
        self.assertIn("formula", answer)
        self.assertIn("step-by-step", answer)
        self.assertIn("final answer", answer)

    def test_casual_non_academic_question_is_rejected_without_math_only_lock(self) -> None:
        user = self.service.register_user("math-only@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="math-core.txt",
            file_bytes=(
                b"Geometry chapter explains triangles, theorem proofs, angle sum property, and congruency criteria. "
                b"Algebra chapter explains linear equations and factorization methods with worked examples."
            ),
        )

        turn = self.service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="Recommend me a good movie for weekend",
        )
        self.assertIn("matric-level teacher only", turn.assistant_message.text.lower())
        self.assertNotIn("mathematics teacher", turn.assistant_message.text.lower())

    def test_equation_style_question_is_understood_from_math_book(self) -> None:
        user = self.service.register_user("equation-understand@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="quadratic-book.txt",
            file_bytes=(
                b"For equation x^2 + 5x + 6 = 0, factorization gives (x+2)(x+3)=0. "
                b"So roots are x = -2 and x = -3. "
                b"This is a solved matric example from chapter quadratic equations."
            ),
        )

        turn = self.service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="x^2 + 5x + 6 = 0 solve",
        )
        answer = turn.assistant_message.text.lower()
        self.assertIn("formula", answer)
        self.assertIn("step-by-step", answer)
        self.assertIn("final answer", answer)

    def test_non_math_subject_question_is_answered_from_uploaded_book(self) -> None:
        user = self.service.register_user("history-subject@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="history.txt",
            file_bytes=(
                b"The Pakistan Resolution was passed on 23 March 1940 at Lahore. "
                b"It became a key milestone in the freedom movement and appears in matric history chapters."
            ),
        )

        turn = self.service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="What is the Pakistan Resolution?",
        )
        answer = turn.assistant_message.text.lower()
        self.assertIn("uploaded material", answer)
        self.assertIn("pakistan resolution", answer)

    def test_rag_generates_question_embedding_for_student_query(self) -> None:
        user = self.service.register_user("rag-embed@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="science.txt",
            file_bytes=(
                b"Photosynthesis converts light energy into glucose in plants. "
                b"This chapter explains chlorophyll, stomata, and oxygen release."
            ),
        )
        with patch.object(self.service, "_generate_embeddings", wraps=self.service._generate_embeddings) as mocked:
            self.service.send_message(
                user_id=user.id,
                conversation_id=convo.id,
                text="Explain photosynthesis in simple words",
            )

        seen_query_embedding = any(
            bool(args)
            and isinstance(args[0], list)
            and len(args[0]) == 1
            and "photosynthesis" in str(args[0][0]).lower()
            for args, _ in mocked.call_args_list
        )
        self.assertTrue(seen_query_embedding)

    def test_rag_retrieves_top_relevant_syllabus_chunks(self) -> None:
        user = self.service.register_user("rag-retrieve@example.com")
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="biology.txt",
            file_bytes=(
                b"Photosynthesis is the process where plants use sunlight, carbon dioxide, and water to make food. "
                b"This topic appears in matric biology chapter one."
            ),
        )
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="history.txt",
            file_bytes=(
                b"Pakistan movement timeline includes key dates, leaders, and constitutional developments. "
                b"This chapter belongs to history syllabus."
            ),
        )

        retrieved = self.service._retrieve_relevant_chunks_scored(
            avatar_id=self.avatar.id,
            question="What is photosynthesis and why is sunlight needed?",
            top_k=3,
            prefer_math=False,
        )
        self.assertGreaterEqual(len(retrieved), 1)
        self.assertIn("photosynthesis", retrieved[0].text.lower())

    def test_rag_uses_strict_system_prompt_for_llm(self) -> None:
        provider = _CapturingLLMProvider()
        service = PersistentChatService(
            repository=self.repo,
            llm_provider=provider,
            config=PersistentServiceConfig(
                max_message_chars=120,
                memory_messages_limit=4,
                rate_limit_max_requests=10,
                rate_limit_window_seconds=60,
            ),
        )
        service.seed_avatars([self.avatar])
        user = service.register_user("rag-prompt@example.com")
        convo = service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="chemistry.txt",
            file_bytes=(
                b"Acids turn blue litmus red and usually have pH below 7 in school chemistry experiments. "
                b"Bases turn red litmus blue and usually have pH above 7. "
                b"This matric chemistry chapter compares indicators, taste, and neutralization reactions."
            ),
        )

        service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="Explain how acids and bases change litmus color.",
        )

        self.assertGreaterEqual(len(provider.calls), 1)
        persona = provider.calls[-1]["persona"].lower()
        prompt = provider.calls[-1]["user_text"].lower()
        self.assertIn("strict retrieval-augmented generation", persona)
        self.assertIn("answer only from retrieved syllabus context", persona)
        self.assertIn("retrieved syllabus context", prompt)

    def test_rag_does_not_use_hard_ooc_rejection_sentence(self) -> None:
        user = self.service.register_user("rag-ooc@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="biology-core.txt",
            file_bytes=(
                b"Photosynthesis and respiration are fundamental processes in plants. "
                b"Students learn chloroplast functions and glucose production."
            ),
        )

        turn = self.service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="Explain Newton's first law of motion",
        )
        answer = turn.assistant_message.text.lower()
        self.assertNotIn("outside your uploaded syllabus context", answer)
        self.assertTrue(
            ("uploaded material" in answer)
            or ("could not find this question clearly" in answer)
        )

    def test_synthesize_audio_truncates_very_long_text_instead_of_failing(self) -> None:
        service = PersistentChatService(
            repository=self.repo,
            config=PersistentServiceConfig(
                max_message_chars=80,
                max_tts_chars=140,
                memory_messages_limit=4,
                rate_limit_max_requests=10,
                rate_limit_window_seconds=60,
            ),
        )
        long_answer = ("Photosynthesis uses sunlight to make glucose. " * 20).strip()
        tts = service.synthesize_audio(text=long_answer, voice_id="alloy")
        payload = tts.audio_bytes.decode("utf-8")
        spoken_text = payload.split("TEXT=", 1)[1]

        self.assertTrue(payload.startswith("VOICE=alloy;TEXT="))
        self.assertLess(len(spoken_text), len(long_answer))
        self.assertLessEqual(len(spoken_text), 205)
        self.assertTrue(spoken_text.endswith("..."))

    def test_science_question_in_natural_wording_is_answered_from_book_context(self) -> None:
        user = self.service.register_user("science-natural@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="science-natural.txt",
            file_bytes=(
                b"Plants prepare their food through photosynthesis using sunlight, carbon dioxide and water. "
                b"Chlorophyll in leaves absorbs light energy for this process in matric science."
            ),
        )

        turn = self.service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="Why do plants make their own food?",
        )
        answer = turn.assistant_message.text.lower()
        self.assertNotIn("matric-level teacher only", answer)
        self.assertIn("uploaded material", answer)

    def test_human_brain_query_matches_paraphrased_science_context(self) -> None:
        user = self.service.register_user("human-brain@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="biology-brain.txt",
            file_bytes=(
                b"Nervous system controls body actions and coordination. "
                b"Neurons carry impulses between spinal cord and organs. "
                b"Cerebrum and cerebellum handle memory and balance in humans."
            ),
        )

        turn = self.service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="tell me about human brain",
        )
        text = turn.assistant_message.text.lower()
        self.assertNotIn("outside your uploaded syllabus context", text)
        self.assertIn("uploaded material", text)

    def test_hinglish_question_is_normalized_to_english_for_retrieval(self) -> None:
        provider = _HinglishTranslationProvider()
        service = PersistentChatService(
            repository=self.repo,
            llm_provider=provider,
            config=PersistentServiceConfig(
                max_message_chars=200,
                memory_messages_limit=4,
                rate_limit_max_requests=10,
                rate_limit_window_seconds=60,
                hinglish_enabled=True,
            ),
        )
        service.seed_avatars([self.avatar])
        user = service.register_user("hinglish-query@example.com")
        convo = service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="biology-hinglish.txt",
            file_bytes=(
                b"Human brain controls memory, thinking, and movement. "
                b"Cerebrum and cerebellum are important parts in matric science chapter."
            ),
        )

        turn = service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="human brain ke bare mein samjhao",
        )
        self.assertIn("answer uploaded books", turn.assistant_message.text.lower())
        self.assertGreaterEqual(len(provider.calls), 2)
        self.assertTrue(any("translation engine" in c["persona"].lower() for c in provider.calls))
        rag_call = next(
            c for c in provider.calls
            if "strict retrieval-augmented generation" in c["persona"].lower()
        )
        self.assertIn("normalized english for retrieval", rag_call["user_text"].lower())
        self.assertIn("explain human brain functions", rag_call["user_text"].lower())

    def test_hinglish_retrieval_query_keeps_original_and_english_context(self) -> None:
        provider = _HinglishTranslationProvider()
        service = PersistentChatService(
            repository=self.repo,
            llm_provider=provider,
            config=PersistentServiceConfig(
                max_message_chars=200,
                memory_messages_limit=4,
                rate_limit_max_requests=10,
                rate_limit_window_seconds=60,
                hinglish_enabled=True,
            ),
        )
        question = "human brain ke bare mein samjhao"
        retrieval_query = service._question_for_retrieval(question)
        lower = retrieval_query.lower()
        self.assertIn("human brain ke bare", lower)
        self.assertIn("explain human brain functions", lower)

    def test_local_teacher_reply_adds_hinglish_suffix(self) -> None:
        user = self.service.register_user("hinglish-local@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        self.service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="chemistry-local.txt",
            file_bytes=(
                b"Acids turn blue litmus red and bases turn red litmus blue in school chemistry chapter. "
                b"This matric lesson explains indicators and neutralization with practical examples."
            ),
        )
        turn = self.service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="What happens to litmus in acid?",
        )
        self.assertIn("Hinglish mein:", turn.assistant_message.text)

    def test_student_image_context_is_appended_once(self) -> None:
        user = self.service.register_user("image-once@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)

        with patch.object(
            self.service,
            "_extract_image_text",
            return_value="The image asks to explain neuron structure with axon and dendrites.",
        ):
            result = self.service.upload_student_image_context(
                user_id=user.id,
                conversation_id=convo.id,
                filename="neuron.png",
                file_bytes=b"fake-image-data",
            )

        self.assertEqual(result.filename, "neuron.png")
        self.assertGreaterEqual(result.extracted_chars, 20)
        self.assertIn("neuron", result.preview.lower())

        loaded = self.service.get_conversation(user_id=user.id, conversation_id=convo.id)
        first = self.service._effective_question(
            user_text="Please answer from image",
            conversation=loaded,
        )
        self.assertIn("Student image context", first)
        self.assertIn("neuron", first.lower())

        second = self.service._effective_question(
            user_text="Please answer from image",
            conversation=loaded,
        )
        self.assertNotIn("Student image context", second)

    def test_student_image_upload_rejects_non_image_file(self) -> None:
        user = self.service.register_user("image-invalid@example.com")
        convo = self.service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        with self.assertRaises(ValidationError):
            self.service.upload_student_image_context(
                user_id=user.id,
                conversation_id=convo.id,
                filename="notes.txt",
                file_bytes=b"not-an-image",
            )

    def test_image_context_is_included_in_rag_prompt(self) -> None:
        provider = _CapturingLLMProvider()
        service = PersistentChatService(
            repository=self.repo,
            llm_provider=provider,
            config=PersistentServiceConfig(
                max_message_chars=200,
                memory_messages_limit=4,
                rate_limit_max_requests=10,
                rate_limit_window_seconds=60,
            ),
        )
        service.seed_avatars([self.avatar])
        user = service.register_user("image-rag@example.com")
        convo = service.create_conversation(user_id=user.id, avatar_id=self.avatar.id)
        service.upload_training_material(
            user_id=user.id,
            avatar_id=self.avatar.id,
            filename="biology-image.txt",
            file_bytes=(
                b"Neuron is the structural and functional unit of nervous system. "
                b"Dendrites receive signals and axon transmits nerve impulses in this chapter."
            ),
        )
        with patch.object(
            service,
            "_extract_image_text",
            return_value="Image question: explain neuron parts and function.",
        ):
            service.upload_student_image_context(
                user_id=user.id,
                conversation_id=convo.id,
                filename="neuron-question.jpg",
                file_bytes=b"fake-image-data",
            )

        service.send_message(
            user_id=user.id,
            conversation_id=convo.id,
            text="Please solve this image question",
        )
        self.assertGreaterEqual(len(provider.calls), 1)
        prompt = provider.calls[-1]["user_text"].lower()
        self.assertIn("student image context", prompt)
        self.assertIn("neuron", prompt)


if __name__ == "__main__":
    unittest.main()
