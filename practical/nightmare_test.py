#!/usr/bin/env python3
"""
Stratum Protocol Bug Test - Nightmare (5 bugs)
==============================================
Tests model's ability to find crypto/protocol-level bugs.

Bugs:
1. Byte order - prev_hash not reversed for block header (Bitcoin convention)
2. Info leak - block_hex logged BEFORE submission (competitors can see)
3. Race/stale - job broadcast vs stale detection race condition
4. Memory leak - recent_shares OrderedDict never pruned
5. Input validation - no hex validation, crash on malformed input

Run: python nightmare_test.py [--api-url URL] [--model MODEL]
"""

import argparse
import json
import re
import requests

PROMPT = '''NIGHTMARE PRACTICAL CODING TEST - STRATUM PROTOCOL & BLOCK VALIDATION

CONTEXT:
This is the stratum mining server and block validation code. After months running 
in production, we've discovered several critical issues:

1. HASH REJECTION: ~5% of valid shares get rejected as "invalid hash" even though
   miners insist they're correct. Affects RandomX algorithm.
2. BLOCK THEFT: When we find a block, sometimes a competitor pool submits it first.
   We suspect they're somehow seeing our blocks before we submit them.
3. STALE WORK: High stale share rate (~8%) even for miners with <50ms latency
4. MEMORY LEAK: Server memory grows ~100MB/day, requires weekly restarts
5. RARE CRASH: Occasional crash when certain miner software connects

Find ALL bugs. Some are subtle cryptographic or protocol-level issues.

### stratum_server.py
```python
import asyncio
import hashlib
import json
import struct
import time
import logging
from typing import Dict, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)

class StratumServer:
    def __init__(self, config):
        self.config = config
        self.clients: Dict[str, StratumClient] = {}
        self.current_job = None
        self.job_counter = 0
        self.recent_shares = OrderedDict()  # For duplicate detection
        self.block_template = None
        
    async def handle_client(self, reader, writer):
        """Handle incoming stratum client connection."""
        addr = writer.get_extra_info('peername')
        client_id = f"{addr[0]}:{addr[1]}"
        client = StratumClient(client_id, reader, writer)
        self.clients[client_id] = client
        
        try:
            while True:
                data = await reader.readline()
                if not data:
                    break
                    
                message = json.loads(data.decode('utf-8'))
                await self._handle_message(client, message)
                
        except Exception as e:
            logger.error(f"Client {client_id} error: {e}")
        finally:
            del self.clients[client_id]
            writer.close()
    
    async def _handle_message(self, client, message):
        """Route stratum messages to handlers."""
        method = message.get('method', '')
        
        if method == 'mining.subscribe':
            await self._handle_subscribe(client, message)
        elif method == 'mining.authorize':
            await self._handle_authorize(client, message)
        elif method == 'mining.submit':
            await self._handle_submit(client, message)
            
    async def _handle_submit(self, client, message):
        """Handle share submission from miner."""
        params = message['params']
        worker, job_id, nonce, ntime, result = params
        
        # Check for duplicate share
        share_key = f"{job_id}:{nonce}:{ntime}"
        if share_key in self.recent_shares:
            await self._send_error(client, message['id'], 'Duplicate share')
            return
            
        self.recent_shares[share_key] = time.time()
        
        # Validate share
        if self._validate_share(job_id, nonce, ntime, result):
            await self._send_result(client, message['id'], True)
            
            # Check if share meets block difficulty
            if self._check_block(job_id, nonce, ntime, result):
                await self._submit_block(job_id, nonce, ntime)
        else:
            await self._send_error(client, message['id'], 'Invalid share')
            
    def _validate_share(self, job_id, nonce, ntime, result) -> bool:
        """Validate submitted share against job difficulty."""
        # Reconstruct block header
        header = self._build_header(job_id, nonce, ntime)
        
        # Hash the header
        hash_result = hashlib.sha256(hashlib.sha256(header).digest()).digest()
        
        # Check against share difficulty
        target = self._get_share_target()
        return int.from_bytes(hash_result, 'little') < target
        
    def _build_header(self, job_id, nonce, ntime) -> bytes:
        """Build block header from components."""
        version = struct.pack('<I', self.block_template['version'])
        prev_hash = bytes.fromhex(self.block_template['previousblockhash'])
        merkle_root = self._compute_merkle(job_id)
        time_bytes = struct.pack('<I', int(ntime, 16))
        bits = bytes.fromhex(self.block_template['bits'])
        nonce_bytes = struct.pack('<I', int(nonce, 16))
        
        return version + prev_hash + merkle_root + time_bytes + bits + nonce_bytes
        
    def _compute_merkle(self, job_id) -> bytes:
        """Compute merkle root from coinbase and merkle branches."""
        coinbase = self.current_job['coinbase']
        merkle = hashlib.sha256(hashlib.sha256(coinbase).digest()).digest()
        
        for branch in self.current_job['merkle_branches']:
            merkle = hashlib.sha256(hashlib.sha256(merkle + bytes.fromhex(branch)).digest()).digest()
            
        return merkle
        
    async def _submit_block(self, job_id, nonce, ntime):
        """Submit found block to network."""
        header = self._build_header(job_id, nonce, ntime)
        block_hex = header.hex() + self.block_template['transactions_hex']
        
        logger.info(f"BLOCK FOUND! Submitting: {block_hex[:64]}...")
        
        # Submit to node
        result = await self._rpc_call('submitblock', [block_hex])
        if result is None:
            logger.info("Block accepted by network!")
        else:
            logger.error(f"Block rejected: {result}")
            
    async def broadcast_job(self, clean=False):
        """Broadcast new mining job to all clients."""
        self.job_counter += 1
        job_id = f"{self.job_counter:08x}"
        
        # Build job parameters
        job = {
            'id': job_id,
            'prevhash': self.block_template['previousblockhash'],
            'coinbase1': self.current_job['coinbase1'],
            'coinbase2': self.current_job['coinbase2'],
            'merkle_branches': self.current_job['merkle_branches'],
            'version': self.block_template['version'],
            'nbits': self.block_template['bits'],
            'ntime': hex(int(time.time()))[2:],
            'clean': clean,
        }
        
        # Send to all clients
        for client in self.clients.values():
            await self._send_job(client, job)
            
    async def _send_result(self, client, msg_id, result):
        message = {'id': msg_id, 'result': result, 'error': None}
        data = json.dumps(message) + '\\n'
        client.writer.write(data.encode())
        await client.writer.drain()
        
    async def _send_error(self, client, msg_id, error):
        message = {'id': msg_id, 'result': None, 'error': error}
        data = json.dumps(message) + '\\n'
        client.writer.write(data.encode())
        await client.writer.drain()
```

TASK:
Find ALL bugs causing:
1. ~5% valid share rejection (hash calculation issue)
2. Block theft by competitors (information leak)
3. High stale rate (job/timing issue)  
4. Memory leak (data structure issue)
5. Rare crash (input validation issue)

For each bug provide:
- Exact location
- Technical root cause  
- Why it causes the specific symptom
- The fix
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
        "byte_order": False,
        "info_leak": False,
        "race_stale": False,
        "memory_leak": False,
        "input_validation": False,
    }
    
    # Bug 1: Byte order / endianness (prev_hash should be reversed)
    byte_patterns = [
        r"prev.*hash.*reverse",
        r"reverse.*prev.*hash",
        r"byte.*order",
        r"endian",
        r"little.*endian",
        r"\[::-1\]",
        r"swap.*bytes",
        r"hash.*backwards",
    ]
    for pattern in byte_patterns:
        if re.search(pattern, response_lower):
            checks["byte_order"] = True
            break
    
    # Bug 2: Info leak (logging block before submission)
    leak_patterns = [
        r"log.*before.*submit",
        r"block.*hex.*log",
        r"logger.*block",
        r"information.*leak",
        r"log.*block.*found",
        r"broadcast.*before",
        r"competitor.*see",
        r"theft.*log",
    ]
    for pattern in leak_patterns:
        if re.search(pattern, response_lower):
            checks["info_leak"] = True
            break
    
    # Bug 3: Race condition / stale shares
    race_patterns = [
        r"race.*condition",
        r"stale.*job",
        r"job.*broadcast",
        r"clean.*job",
        r"active.*job",
        r"job.*validation",
        r"old.*job",
    ]
    for pattern in race_patterns:
        if re.search(pattern, response_lower):
            checks["race_stale"] = True
            break
    
    # Bug 4: Memory leak (OrderedDict never pruned)
    memory_patterns = [
        r"ordereddict.*never.*prune",
        r"recent.*shares.*grow",
        r"memory.*leak",
        r"never.*clean",
        r"unbounded.*growth",
        r"prune.*recent",
        r"clear.*old.*shares",
    ]
    for pattern in memory_patterns:
        if re.search(pattern, response_lower):
            checks["memory_leak"] = True
            break
    
    # Bug 5: Input validation (hex parsing crash)
    input_patterns = [
        r"hex.*valid",
        r"input.*valid",
        r"malform",
        r"try.*except",
        r"crash.*invalid",
        r"parse.*error",
        r"valueerror",
    ]
    for pattern in input_patterns:
        if re.search(pattern, response_lower):
            checks["input_validation"] = True
            break
    
    score = sum(1 for v in checks.values() if v)
    return {
        "score": score,
        "max_score": 5,
        "passed": score >= 5,
        "checks": checks,
    }

def main():
    parser = argparse.ArgumentParser(description="Nightmare Stratum Bug Test")
    parser.add_argument('--api-url', default='http://localhost:8000/v1')
    parser.add_argument('--model', default='seed-oss')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()
    
    print("=" * 60)
    print("STRATUM PROTOCOL BUG TEST (Nightmare - 5 bugs)")
    print("=" * 60)
    
    try:
        r = requests.post(
            f"{args.api_url}/chat/completions",
            json={
                'model': args.model,
                'messages': [{'role': 'user', 'content': PROMPT}],
                'max_tokens': 10000,
                'temperature': 0.0
            },
            timeout=args.timeout
        )
        data = r.json()
        content = data['choices'][0]['message']['content']
        clean = strip_thinking(content)
        
        print("\n=== RESPONSE ===\n")
        print(clean[:5000] + "..." if len(clean) > 5000 else clean)
        
        print("\n=== EVALUATION ===")
        result = evaluate(clean)
        
        for check, passed in result['checks'].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print(f"\nScore: {result['score']}/{result['max_score']}")
        if result['passed']:
            print("🎉 EXCELLENT - All nightmare bugs found!")
        elif result['score'] >= 4:
            print("✓ VERY GOOD - Found most bugs")
        elif result['score'] >= 3:
            print("△ GOOD - Solid analysis")
        else:
            print(f"✗ NEEDS WORK - Found {result['score']}/5 bugs")
            
        print(f"\nTokens: {data.get('usage', {})}")
        
        return result
        
    except Exception as e:
        print(f"ERROR: {e}")
        return {"score": 0, "max_score": 5, "passed": False}

if __name__ == "__main__":
    main()
