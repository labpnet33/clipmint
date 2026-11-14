"""Regression tests for conservative, transcript-only clip completion."""
import unittest

from shorts_generator.viral_moments import (
    complete_punchline_end, reaction_cue, score_moment_integrity,
    unfinished_phrase,
)


class ViralMomentTests(unittest.TestCase):
    def setUp(self):
        self.segments = [
            {"start": 0.0, "end": 7.0, "text": "I knew what was happening."},
            {"start": 7.1, "end": 10.0, "text": "But then I said because"},
            {"start": 10.2, "end": 12.4, "text": "Nobody believed the real answer!"},
            {"start": 12.5, "end": 13.5, "text": "[laughter]"},
        ]

    def test_unfinished_clauses(self):
        self.assertTrue(unfinished_phrase("And I said but"))
        self.assertTrue(unfinished_phrase("Aur maine kaha ki"))
        self.assertTrue(unfinished_phrase("And then..."))
        self.assertFalse(unfinished_phrase("But I said no!"))
        self.assertFalse(unfinished_phrase("Yeh toh kamaal hai."))

    def test_transcripted_reaction(self):
        self.assertTrue(reaction_cue("[laughter]"))
        self.assertTrue(reaction_cue("applause"))
        self.assertFalse(reaction_cue("the crowd watched carefully"))

    def test_rescue_incomplete_ending(self):
        end, notes = complete_punchline_end(self.segments, 0, 10.0, 13.0)
        self.assertEqual(end, 12.4)
        self.assertIn("unfinished", notes[0])

    def test_keep_reaction_after_payoff(self):
        end, notes = complete_punchline_end(self.segments, 0, 12.4, 14.0)
        self.assertEqual(end, 13.5)
        self.assertIn("reaction", notes[0])

    def test_respects_length_ceiling(self):
        end, notes = complete_punchline_end(self.segments, 0, 10.0, 9.0)
        self.assertEqual(end, 10.0)
        self.assertFalse(notes)

    def test_skips_unrelated_followup(self):
        segs = [dict(s) for s in self.segments]
        segs[2]["start"] = 15.0
        segs[2]["end"] = 16.0
        end, notes = complete_punchline_end(segs, 0, 10.0, 25.0)
        self.assertEqual(end, 10.0)
        self.assertFalse(notes)

    def test_no_hallucinated_reaction(self):
        segs = [dict(s) for s in self.segments]
        segs[1]["text"] = "It was a beautiful day."
        segs[2]["text"] = "I went home afterwards."
        end, _ = complete_punchline_end(segs, 0, 10.0, 25.0)
        self.assertEqual(end, 10.0)

    def test_no_mid_segment_breakage(self):
        d = score_moment_integrity({"start_time": 0, "end_time": 8},
                                   {"segments": self.segments})
        self.assertEqual(d["adjustment"], -10)

    def test_downgrade_unfinished_ending(self):
        d = score_moment_integrity({"start_time": 0, "end_time": 10},
                                   {"segments": self.segments})
        self.assertEqual(d["adjustment"], -8)

    def test_reaction_reward(self):
        d = score_moment_integrity({"start_time": 0, "end_time": 13.5},
                                   {"segments": self.segments})
        self.assertEqual(d["adjustment"], 3)

    def test_missing_transcript_is_neutral(self):
        d = score_moment_integrity({"start_time": 0, "end_time": 10}, None)
        self.assertEqual(d["adjustment"], 0)

    def test_complete_statement_is_neutral(self):
        d = score_moment_integrity({"start_time": 0, "end_time": 7},
                                   {"segments": self.segments})
        self.assertEqual(d["adjustment"], 0)

    def test_no_extension_of_another_unfinished_line(self):
        segs = [dict(s) for s in self.segments]
        segs[2]["text"] = "because we wanted to"
        end, notes = complete_punchline_end(segs, 0, 10, 16)
        self.assertEqual(end, 10)
        self.assertFalse(notes)


if __name__ == "__main__":
    unittest.main()
