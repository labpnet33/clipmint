"""AI Activity tests. All AI backends mocked; no cloud calls, GPU, or downloads."""
import unittest
from unittest.mock import Mock, patch
from shorts_generator.local import ai_activity as a


class ActivityTests(unittest.TestCase):
    def test_records_real_provider_and_timing(self):
        events = []
        wrapped = a.trace_provider('ollama')(lambda prompt: '{"ok":true}')
        with a.watch('job-1', events.append), a.stage('seo'):
            self.assertIn('ok', wrapped('do work'))
        self.assertEqual([e['status'] for e in events], ['running', 'success'])
        self.assertEqual(events[0]['id'], events[1]['id'])
        self.assertEqual(events[-1]['provider'], 'ollama')
        self.assertEqual(events[-1]['stage'], 'seo')
        self.assertGreaterEqual(events[-1]['duration_ms'], 0)

    def test_no_request_content_or_secrets_in_activity(self):
        events = []
        def fail(_prompt):
            raise RuntimeError('secret: API-KEY-VERY-PRIVATE text transcript')
        with a.watch('j', events.append), a.stage('highlight'):
            with self.assertRaises(RuntimeError):
                a.trace_provider('groq')(fail)('private transcript content')
        text = str(events)
        self.assertNotIn('API-KEY-VERY-PRIVATE', text)
        self.assertNotIn('private transcript content', text)
        self.assertEqual(events[-1]['error'], 'RuntimeError')

    def test_does_not_swallow_provider_error(self):
        def fail():
            raise ValueError('test')
        with a.watch('job', lambda *_: None):
            with self.assertRaises(ValueError):
                a.trace_provider('gemini')(fail)()

    def test_without_job_no_telemetry_and_no_behavior_change(self):
        called = Mock(return_value='ok')
        self.assertEqual(a.trace_provider('gemini')(called)(), 'ok')
        called.assert_called_once()

    def test_stage_wrapper_isolates_tasks(self):
        events = []
        original = a.trace_provider('openai')(lambda: 'done')
        with a.watch('one', events.append):
            a.for_stage('highlight', original)()
            a.for_stage('seo', original)()
        self.assertEqual([e['stage'] for e in events if e['status']=='success'], ['highlight', 'seo'])

    def test_fallback_detects_actual_backend(self):
        events = []
        with patch('shorts_generator.config.current_stage_provider', return_value='gemini'):
            with a.watch('j', events.append), a.stage('highlight'):
                a.trace_provider('groq')(lambda: 1)()
        self.assertTrue(events[-1]['fallback'])

    def test_transcription_distinct_from_cloud_request_usage(self):
        events = []
        with a.watch('j', events.append):
            a.trace_transcription(lambda filename: {'segments':[1]})('input.mp4')
        self.assertEqual(events[-1]['stage'], 'transcribe')
        self.assertEqual(a.summarize(events[1:])['cloud_requests'], 0)
        self.assertEqual(a.summarize(events[1:])['requests'], 0)

    def test_sequential_jobs_isolated(self):
        one, two = [], []
        req = a.trace_provider('ollama')(lambda: None)
        with a.watch('one', one.append): req()
        with a.watch('two', two.append): req()
        self.assertNotEqual(one[-1]['id'], two[-1]['id'])

    def test_nested_stage_restored(self):
        events=[]
        fn=a.trace_provider('gemini')(lambda: 'x')
        with a.watch('j',events.append), a.stage('highlight'):
            a.for_stage('seo',fn)()
            fn()
        self.assertEqual([e['stage'] for e in events if e['status']=='success'],['seo','highlight'])

    def test_http_code_reveals_no_message(self):
        class HttpError(Exception):
            status_code = 429
        err=HttpError('Bearer secret')
        self.assertEqual(a._safe_code(err), 'HttpError (HTTP 429)')

if __name__ == '__main__':
    unittest.main()
