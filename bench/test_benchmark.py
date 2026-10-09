"""Offline tests: python -m unittest discover -s bench -p 'test_benchmark.py'."""
import argparse
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import benchmark


def event(value):
    return "data: " + json.dumps(value) + "\n\n"


def delta(**values):
    return event({"choices": [{"delta": values}]})


def usage(count=4):
    return event({"choices": [{"delta": {}, "finish_reason": "length"}]}) + event(
        {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": count,
         "completion_tokens_details": {"reasoning_tokens": 2}}})


class Response(io.BytesIO):
    status = 200


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.args = argparse.Namespace(seed=42, model="qwen3.8-27b", prompt_repeat=1,
                                       max_tokens=4, thinking=False, allow_eos=False)
        self.payload = benchmark.make_payload(self.args, 0)

    def run_stream(self, text, times=None):
        clock = iter(times or range(100)).__next__
        return benchmark.run_request("http://localhost/v1", self.payload,
                                     opener=lambda *a, **kw: Response(text.encode()), clock=clock)

    def test_role_content_reasoning_and_usage(self):
        stream = delta(role="assistant") + delta(reasoning_content="think") + delta(content="code") + usage() + "data: [DONE]\n\n"
        result = self.run_stream(stream, [0, 2, 4, 5])
        self.assertTrue(result["success"])
        self.assertEqual(result["ttft_s"], 2)
        self.assertEqual(result["latency_s"], 5)
        self.assertEqual(result["decode_tokens_per_s_approx"], 2)
        self.assertEqual(result["reasoning"], "think")
        self.assertEqual(result["content"], "code")
        self.assertEqual(result["reasoning_tokens"], 2)
        self.assertEqual(result["raw_sse"], stream)
        self.assertEqual(result["payload_sha256"], hashlib.sha256(result["request_body"].encode()).hexdigest())

    def test_reasoning_alias_and_single_chunk(self):
        result = self.run_stream(delta(reasoning="abcd") + usage() + "data: [DONE]\n\n")
        self.assertTrue(result["success"])
        self.assertIsNone(result["decode_tokens_per_s_approx"])

    def test_failure_cases(self):
        cases = {
            "empty output": delta(role="assistant") + usage() + "data: [DONE]\n\n",
            "missing usage": delta(content="hi") + "data: [DONE]\n\n",
            "missing [DONE]": delta(content="hi") + usage(),
            "short output": delta(content="hi") + usage(2) + "data: [DONE]\n\n",
            "exceeds max_tokens": delta(content="hi") + usage(5) + "data: [DONE]\n\n",
            "finish_reason": delta(content="hi") + event({"usage": {"prompt_tokens": 20, "completion_tokens": 4}}) + "data: [DONE]\n\n",
            "invalid usage": delta(content="hi") + usage("4") + "data: [DONE]\n\n",
            "server error": event({"error": {"message": "bad request"}}),
            "JSONDecodeError": "data: not json\n\n",
            "missing blank line": delta(content="hi") + "data: [DONE]\n",
        }
        for error, stream in cases.items():
            with self.subTest(error=error):
                result = self.run_stream(stream)
                self.assertFalse(result["success"])
                self.assertIn(error, " ".join(result["errors"]))

    def test_concurrency_trial_seeds_avoid_prefix_reuse(self):
        # trial-spark.sh uses 42 + concurrency for independent cold batches.
        self.args.seed = 43
        c1 = benchmark.make_payload(self.args, 0)
        self.args.seed = 46
        c4 = benchmark.make_payload(self.args, 0)
        self.assertNotEqual(c1["messages"][0]["content"].splitlines()[0],
                            c4["messages"][0]["content"].splitlines()[0])
        self.assertEqual(c4, benchmark.make_payload(self.args, 0))

    def test_allow_eos(self):
        self.payload["ignore_eos"] = False
        result = self.run_stream(delta(content="hi") + usage(2) + "data: [DONE]\n\n")
        self.assertTrue(result["success"])

    def test_http_error_persisted(self):
        def opener(*a, **kw):
            raise urllib.error.HTTPError("url", 400, "bad", {}, io.BytesIO(b"details"))
        result = benchmark.run_request("http://localhost/v1", self.payload, opener=opener)
        self.assertFalse(result["success"])
        self.assertEqual(result["http_status"], 400)
        self.assertEqual(result["http_error_body"], "details")

    def test_sse_comments_multiline(self):
        stream = ': heartbeat\r\ndata: {"choices":\r\ndata: []}\r\n\r\n'
        raw = []
        self.assertEqual(list(benchmark.sse_events(Response(stream.encode()), raw)), ['{"choices":\n[]}'])

    def test_prompts_reproducible_distinct_and_long(self):
        self.assertEqual(self.payload, benchmark.make_payload(self.args, 0))
        warmup = benchmark.make_payload(self.args, 0, warmup=True)
        second = benchmark.make_payload(self.args, 1)
        beginnings = {p["messages"][0]["content"].splitlines()[0] for p in (self.payload, warmup, second)}
        self.assertEqual(len(beginnings), 3)
        self.args.prompt_repeat = 3
        self.assertGreater(len(benchmark.encode_payload(benchmark.make_payload(self.args, 0))), len(benchmark.encode_payload(self.payload)))

    def test_cli_persists_warmup_and_failed_measurements(self):
        streams = [delta(content="ok") + usage() + "data: [DONE]\n\n",
                   delta(content="short") + usage(1) + "data: [DONE]\n\n"]
        with tempfile.TemporaryDirectory() as tmp, patch.object(benchmark.urllib.request, "urlopen", side_effect=lambda *a, **kw: Response(streams.pop(0).encode())):
            target = str(Path(tmp) / "out.json")
            status = benchmark.main(["--base-url", "http://localhost/v1", "--label", "test", "--output", target,
                                     "--requests", "1", "--max-tokens", "4"])
            report = json.loads(Path(target).read_text())
        self.assertEqual(status, 1)
        self.assertTrue(report["warmup"]["success"])
        self.assertEqual(report["summary"]["requests"], 1)
        self.assertEqual(report["summary"]["failed"], 1)
        self.assertFalse(report["success"])

    def test_percentiles(self):
        self.assertEqual(benchmark.percentiles([1, 3])["p50"], 2)
        self.assertIsNone(benchmark.percentiles([])["p99"])


if __name__ == "__main__":
    unittest.main()
