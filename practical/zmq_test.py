#!/usr/bin/env python3
"""
ZMQ Listener Bug Test - Easy (1 bug)
====================================
Tests model's ability to find a threading bug in ZMQ code.

Bug: Socket created in main thread but used in background thread.
ZMQ sockets are not thread-safe across threads.

Run: python zmq_test.py [--api-url URL] [--model MODEL]
"""

import argparse
import json
import re
import requests

PROMPT = '''PRACTICAL CODING TEST - ZMQ LISTENER BUG

CONTEXT:
Our mining pool uses ZeroMQ to receive block notifications from the node.
The ZMQ listener thread starts but never receives any messages, even though 
the node is correctly publishing to the configured endpoint.

The listener works when tested standalone but fails when integrated into the pool.

### zmq_listener.py
```python
import zmq
import threading
import logging

logger = logging.getLogger(__name__)

class ZMQListener:
    def __init__(self, endpoint: str, callback):
        self.endpoint = endpoint
        self.callback = callback
        self.running = False
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
    def start(self):
        """Start the ZMQ listener in a background thread."""
        self.running = True
        self.socket.connect(self.endpoint)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "hashblock")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "rawblock")
        
        thread = threading.Thread(target=self._listen, daemon=True)
        thread.start()
        logger.info(f"ZMQ listener started on {self.endpoint}")
        
    def _listen(self):
        """Main listener loop - runs in background thread."""
        while self.running:
            try:
                topic = self.socket.recv_string()
                body = self.socket.recv()
                logger.debug(f"Received ZMQ message: {topic}")
                self.callback(topic, body)
            except zmq.ZMQError as e:
                logger.error(f"ZMQ error: {e}")
                break
                
    def stop(self):
        """Stop the listener."""
        self.running = False
        self.socket.close()
        self.context.term()
```

### pool_server.py (usage)
```python
from zmq_listener import ZMQListener

class PoolServer:
    def __init__(self, config):
        self.config = config
        self.zmq = ZMQListener(
            config['zmq_endpoint'],
            self._handle_block
        )
        
    def start(self):
        # Start ZMQ listener
        self.zmq.start()
        
        # Start stratum server (blocks forever)
        self._run_stratum()
        
    def _handle_block(self, topic, body):
        logger.info(f"New block received via ZMQ: {topic}")
        # Process block notification...
```

TASK:
Find the bug that causes the ZMQ listener to never receive messages.
Explain the root cause and provide a fix.
'''

def strip_thinking(content: str) -> str:
    """Remove model thinking tags."""
    content = re.sub(r'<seed:think>.*?</seed:think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<seed:[^>]+>', '', content)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return content.strip()

def evaluate(response: str) -> dict:
    """Evaluate response for bug detection."""
    response_lower = response.lower()
    
    checks = {
        "socket_thread_issue": False,
        "context_thread_safety": False,
    }
    
    # Check 1: Socket created in main thread, used in different thread
    thread_patterns = [
        r"main.*thread",
        r"socket.*creat.*different.*thread",
        r"socket.*thread",
        r"creat.*socket.*start",
        r"__init__.*thread",
    ]
    for pattern in thread_patterns:
        if re.search(pattern, response_lower):
            checks["socket_thread_issue"] = True
            break
    
    # Check 2: ZMQ context/socket thread safety
    safety_patterns = [
        r"context.*thread",
        r"not.*thread.*safe",
        r"zmq.*socket.*thread",
        r"share.*socket",
        r"socket.*not.*share",
    ]
    for pattern in safety_patterns:
        if re.search(pattern, response_lower):
            checks["context_thread_safety"] = True
            break
    
    score = sum(1 for v in checks.values() if v)
    return {
        "score": score,
        "max_score": 1,  # Only need to find 1 bug
        "passed": score >= 1,
        "checks": checks,
    }

def main():
    parser = argparse.ArgumentParser(description="ZMQ Bug Test")
    parser.add_argument('--api-url', default='http://localhost:8000/v1')
    parser.add_argument('--model', default='seed-oss')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()
    
    print("=" * 60)
    print("ZMQ LISTENER BUG TEST (Easy)")
    print("=" * 60)
    
    try:
        r = requests.post(
            f"{args.api_url}/chat/completions",
            json={
                'model': args.model,
                'messages': [{'role': 'user', 'content': PROMPT}],
                'max_tokens': 4096,
                'temperature': 0.0
            },
            timeout=args.timeout
        )
        data = r.json()
        content = data['choices'][0]['message']['content']
        clean = strip_thinking(content)
        
        print("\n=== RESPONSE ===\n")
        print(clean[:2000] + "..." if len(clean) > 2000 else clean)
        
        print("\n=== EVALUATION ===")
        result = evaluate(clean)
        
        for check, passed in result['checks'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print(f"\nScore: {result['score']}/{result['max_score']}")
        if result['passed']:
            print("✓ PASSED - Bug found!")
        else:
            print("✗ FAILED - Bug not identified")
            
        print(f"\nTokens: {data.get('usage', {})}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {"score": 0, "max_score": 1, "passed": False}

if __name__ == "__main__":
    main()
