"""Regression tests for ClipMint SEO generation and Rewrite behaviour.

Run: python -m unittest discover -s tests -p test_seo_quality.py -v
No external API calls or video files needed.
"""
import unittest
from unittest.mock import patch

from shorts_generator import seo


class SeoQualityTests(unittest.TestCase):
    def test_rejects_unwritten_ai_metadata(self):
        h = {"title": "Original", "hook_sentence": "A real joke", "scene": {}}
        self.assertFalse(seo._usable_metadata({"index": 1}, h, {}))
        self.assertFalse(seo._usable_metadata({"index": 1, "title": "Hello"}, h, {}))
        self.assertFalse(seo._usable_metadata({"title":"Valid title", "description_options":[{"description": None}]}, h, {}))
        self.assertTrue(seo._usable_metadata({
            "index": 1, "title": "A Real Joke About Something Funny",
            "description": "A real joke from the comedy performance."}, h, {}))
        self.assertFalse(seo._usable_metadata({
            "title_options": [{"title": "Rejected false claim", "truth": 1}],
            "description": "A description"}, h, {}))

    def test_duplicate_title_picks_alternate_for_rewrite(self):
        entries = [{"title": "Same Joke About The Big Surprise",
                    "title_options": [
                        {"title": "Same Joke About The Big Surprise"},
                        {"title": "A New Angle On The Real Joke"}],
                    "generated": True}]
        seo._spread_leads(entries, avoid_titles=["Same joke about the big surprise!"])
        self.assertEqual(entries[0]["title"], "A New Angle On The Real Joke")
        self.assertTrue(entries[0]["generated"])

    def test_unchanged_rewrite_cannot_claim_success(self):
        entries = [{"title": "Same Joke About The Big Surprise",
                    "title_options": [{"title": "Same Joke About The Big Surprise"}],
                    "generated": True}]
        seo._spread_leads(entries, avoid_titles=["Same Joke About The Big Surprise"])
        self.assertFalse(entries[0]["generated"])

    def test_batch_deduplication(self):
        entries = [
            {"title": "Funny Big Moment From The Stream",
             "title_options": [{"title": "Funny Big Moment From The Stream"}], "generated": True},
            {"title": "Funny Big Moment From The Stream",
             "title_options": [{"title": "A Different Detail In This Clip"}], "generated": True},
        ]
        seo._spread_leads(entries)
        self.assertNotEqual(entries[0]["title"], entries[1]["title"])

    def test_rewrite_avoids_own_previous_title(self):
        # Exercise the same regenerate_seo() that the UI's Rewrite button calls.
        import threading
        from unittest.mock import Mock
        from webapp import jobs

        clip = {"file": "one.mp4", "seo": {"title": "My Previous Title"},
                "scene": {"genre": "comedy"}, "start_time": 1, "end_time": 12}
        other = {"file": "two.mp4", "seo": {"title": "Other Clip Title"}}
        job = Mock()
        job.clips = [clip, other]
        job.video_meta = {}
        job.source = "video.mp4"
        job.subject = {"content_kind": "other"}
        job.source_path = None
        job.out_dir = "ignored-by-mock"
        store = Mock()
        store._lock = threading.RLock()
        store.set_seo.return_value = clip
        result = [{"title": "A Completely Different Funny Moment",
                   "description": "This joke is completely different.", "generated": True}]
        with patch('webapp.jobs._clip_words', return_value='A joke happens'), \
             patch('shorts_generator.seo.generate_seo', return_value=result) as gen, \
             patch('webapp.jobs.rename_to_title', return_value='one.mp4'):
            count = jobs.regenerate_seo(store, job, force=True, only='one.mp4')
        self.assertEqual(count, 1)
        excluded = gen.call_args.kwargs['avoid_titles']
        self.assertIn('My Previous Title', excluded)
        self.assertIn('Other Clip Title', excluded)
        from shorts_generator.local.task_routing import call_seo_llm
        self.assertEqual((gen.call_args.kwargs['llm_fn'].__module__, gen.call_args.kwargs['llm_fn'].__qualname__), (call_seo_llm.__module__, call_seo_llm.__qualname__))

    @patch.object(seo.playbook, 'prompt_text', return_value='playbook')
    @patch.object(seo.playbook, 'load', return_value=('playbook', 'local'))
    def test_initial_generation_retries_incomplete_and_uses_hinglish_instruction(self, *_):
        calls = []

        def fake_llm(prompt):
            calls.append(prompt)
            if 'LANGUAGE, AUDIENCE AND COMEDY' in prompt and len([p for p in calls if 'LANGUAGE, AUDIENCE AND COMEDY' in p]) == 1:
                return '{"clips":[{"index":1,"title_options":[]}]}'
            if 'LANGUAGE, AUDIENCE AND COMEDY' in prompt:
                return '{"clips":[{"index":1,"title":"Samay Ka Yeh Joke Sunke Sab Hans Pade", "description":"An actual comedy moment from a standup set."}]}'
            return '{"clips":[]}'

        highlights = [{"title": "Hook", "hook_sentence": "A joke",
                       "start_time": 0, "end_time": 10,
                       "transcript_text": "Maine kaha yeh kya hua", "scene": {"genre": "standup"}}]
        # Test the cloud prompt explicitly, regardless of the selected Ollama setting.
        # The LLM response is mocked: this does NOT call Gemini or spend API quota.
        with patch("shorts_generator.config.current_provider", return_value="gemini"), \
             patch("shorts_generator.config.current_stage_provider", return_value="gemini"):
            out = seo.generate_seo(highlights, subject={}, llm_fn=fake_llm)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]['generated'])
        self.assertIn('Samay Ka Yeh Joke', out[0]['title'])
        self.assertTrue(any('natural conversational Hinglish' in p for p in calls))
        self.assertGreaterEqual(sum('LANGUAGE, AUDIENCE AND COMEDY' in p for p in calls), 2)


if __name__ == '__main__':
    unittest.main()
