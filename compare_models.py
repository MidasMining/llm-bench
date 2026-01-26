#!/usr/bin/env python3
"""
Model Comparison Test Harness v2.0
==================================
Complete benchmark suite for comparing local LLM models.

Includes:
- Standard benchmarks (46 tests): code, reasoning, knowledge, tool use, speed, context
- Practical debugging tests: zmq, pplns, expert, nightmare, hiveos_wrapper
- Long context test: multi-file codebase analysis

Usage:
    # Full comparison (sequential - one model at a time)
    python compare_models.py --config models.yaml
    
    # Quick comparison (skip slow tests)
    python compare_models.py --config models.yaml --quick
    
    # Run specific model only
    python compare_models.py --model seed-oss --api-url http://localhost:8000/v1
    
    # Run specific test categories
    python compare_models.py --tests practical nightmare

Author: Generated for Midas's A4000 inference setup
Date: January 2026
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests

VERSION = "2.0"

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    "models": [
        {
            "name": "Seed-OSS-36B",
            "api_url": "http://localhost:8000/v1",
            "model_id": "seed-oss",
            "thinking_tag": "<seed:think>",
        }
    ],
    "tests": {
        "standard": True,
        "practical": True,
        "long_context": True,
    },
    "settings": {
        "temperature": 0.0,
        "max_tokens": 8192,
        "timeout": 600,
    }
}

# ============================================================================
# COLORS
# ============================================================================

class Colors:
    HEADER = '\033[1;93m'
    PASS = '\033[92m'
    FAIL = '\033[91m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

C = Colors()

# ============================================================================
# PRACTICAL TEST PROMPTS
# ============================================================================

PRACTICAL_TESTS = {
    "zmq": {
        "name": "ZMQ Listener Bug",
        "difficulty": "Easy",
        "expected_bugs": 1,
        "description": "Single threading bug causing listener to never run",
        "prompt": '''PRACTICAL CODING TEST - ZMQ LISTENER BUG

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
''',
        "checks": [
            ("socket creation in main thread", ["main thread", "socket", "different thread", "thread", "context"]),
            ("zmq context not thread safe this way", ["context", "thread", "not.*safe", "share"]),
        ]
    },
    
    "pplns": {
        "name": "PPLNS Mining Pool Bugs",
        "difficulty": "Hard",
        "expected_bugs": 3,
        "description": "Config mismatch, hardcoded values, method errors",
        "prompt": '''PRACTICAL CODING TEST - PPLNS MINING POOL BUGS

CONTEXT:
Our PPLNS mining pool has several issues that are affecting miner payouts:
1. Miners complain their shares aren't being counted properly
2. Payouts are sometimes calculated incorrectly  
3. The share window seems wrong

### pplns.py
```python
import sqlite3
import time
import logging
from decimal import Decimal

logger = logging.getLogger(__name__)

class PPLNSManager:
    def __init__(self, config, db_path: str):
        self.config = config
        self.db_path = db_path
        self.n_factor = config.get('pplns_n', 2)  # N in PPLNS
        self.share_window = 10000  # Last N shares to consider
        
    def record_share(self, worker: str, difficulty: float):
        """Record a share from a worker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(\'\'\'
            INSERT INTO shares (worker, difficulty, timestamp)
            VALUES (?, ?, ?)
        \'\'\', (worker, difficulty, time.time()))
        
        conn.commit()
        conn.close()
        
    def calculate_payouts(self, block_reward: Decimal) -> dict:
        """Calculate PPLNS payouts for a found block."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get network difficulty for window calculation
        net_diff = self.config.get('network_difficulty', 1000000)
        window_size = int(net_diff * self.n_factor)
        
        # Get recent shares within window
        cursor.execute(\'\'\'
            SELECT worker, SUM(difficulty) as total_diff
            FROM shares
            WHERE id > (SELECT MAX(id) - ? FROM shares)
            GROUP BY worker
        \'\'\', (self.share_window,))
        
        shares = cursor.fetchall()
        conn.close()
        
        if not shares:
            logger.warning("No shares found for payout calculation")
            return {}
        
        # Calculate total difficulty in window
        total_diff = sum(s[1] for s in shares)
        
        # Calculate each worker's payout
        payouts = {}
        for worker, worker_diff in shares:
            share_pct = Decimal(str(worker_diff)) / Decimal(str(total_diff))
            payout = block_reward * share_pct
            payouts[worker] = float(payout)
            
        logger.info(f"Calculated payouts for {len(payouts)} workers")
        return payouts
        
    def get_worker_stats(self, worker: str) -> dict:
        """Get statistics for a specific worker."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get total shares
        cursor.execute(
            'SELECT COUNT(*), SUM(difficulty) FROM shares WHERE worker = ?',
            (worker,)
        )
        count, total_diff = cursor.fetchone()
        
        # Get shares in current window
        cursor.execute(\'\'\'
            SELECT COUNT(*), SUM(difficulty) FROM shares 
            WHERE worker = ? AND id > (SELECT MAX(id) - ? FROM shares)
        \'\'\', (worker, self.share_window))
        window_count, window_diff = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_shares': count or 0,
            'total_difficulty': total_diff or 0,
            'window_shares': window_count or 0,
            'window_difficulty': window_diff or 0,
        }

    def get_pool_hashrate(self) -> float:
        """Calculate pool hashrate from recent shares."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get shares from last 10 minutes
        ten_min_ago = time.time() - 600
        cursor.execute(\'\'\'
            SELECT SUM(difficulty), MIN(timestamp), MAX(timestamp)
            FROM shares WHERE timestamp > ?
        \'\'\', (ten_min_ago,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result or not result[0]:
            return 0.0
            
        total_diff, min_time, max_time = result
        time_span = max_time - min_time
        
        if time_span <= 0:
            return 0.0
            
        # Hashrate = difficulty / time (simplified)
        return total_diff / time_span
```

### pool_config.json
```json
{
    "pool_name": "ThoughtPool",
    "pplns_window": 50000,
    "pplns_n": 2,
    "network_difficulty": 5000000,
    "block_reward": 78.5,
    "stratum_port": 3333
}
```

TASK:
Find ALL bugs affecting share counting and payout calculations.
For each bug provide:
1. Exact location (file, line/method)
2. Root cause
3. The fix
''',
        "checks": [
            ("hardcoded share_window ignores config", ["hardcoded", "10000", "share_window", "config", "pplns_window"]),
            ("window_size calculated but not used", ["window_size", "not used", "calculated.*never", "unused"]),
            ("config key mismatch", ["pplns_window", "pplns_n", "config", "mismatch", "key"]),
        ]
    },
    
    "expert": {
        "name": "Payment System Bugs",
        "difficulty": "Expert",
        "expected_bugs": 4,
        "description": "Race conditions, SQL injection, float precision, atomicity",
        "prompt": '''EXPERT PRACTICAL CODING TEST - PAYMENT SYSTEM BUGS

CONTEXT:
Our mining pool payment system has critical bugs causing:
1. Occasional duplicate payments to the same address
2. Security vulnerability reported by auditor
3. Small discrepancies in payment amounts (fractions of coins disappearing)
4. Payments sometimes fail silently

### payment_processor.py
```python
import sqlite3
import threading
import time
import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)

class PaymentProcessor:
    def __init__(self, db_path: str, rpc_url: str):
        self.db_path = db_path
        self.rpc_url = rpc_url
        self.processing = set()  # Track payments being processed
        self.lock = threading.Lock()
        
    def process_pending_payments(self):
        """Process all pending payments."""
        pending = self._get_pending_payouts()
        
        for payout in pending:
            payout_id, address, amount = payout
            
            # Check if already processing
            if address in self.processing:
                continue
                
            self.processing.add(address)
            
            try:
                self._execute_payment(payout_id, address, amount)
            finally:
                self.processing.discard(address)
                
    def _get_pending_payouts(self) -> list:
        """Get list of pending payouts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(\'\'\'
            SELECT id, address, amount FROM payout_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
        \'\'\')
        
        results = cursor.fetchall()
        conn.close()
        return results
        
    def _execute_payment(self, payout_id: int, address: str, amount: float):
        """Execute a single payment."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Mark as processing
            cursor.execute(
                f"UPDATE payout_queue SET status = 'processing' WHERE id = {payout_id}"
            )
            conn.commit()
            
            # Send payment via RPC
            txid = self._send_payment(address, amount)
            
            if txid:
                # Mark as completed
                cursor.execute(\'\'\'
                    UPDATE payout_queue 
                    SET status = 'completed', txid = ?, completed_at = ?
                    WHERE id = ?
                \'\'\', (txid, time.time(), payout_id))
                conn.commit()
                logger.info(f"Payment {payout_id} completed: {txid}")
            else:
                # Mark as failed
                cursor.execute(\'\'\'
                    UPDATE payout_queue SET status = 'failed' WHERE id = ?
                \'\'\', (payout_id,))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Payment {payout_id} error: {e}")
            cursor.execute(\'\'\'
                UPDATE payout_queue SET status = 'pending' WHERE id = ?
            \'\'\', (payout_id,))
            conn.commit()
        finally:
            conn.close()
            
    def _send_payment(self, address: str, amount: float) -> Optional[str]:
        """Send payment via node RPC."""
        try:
            response = requests.post(self.rpc_url, json={
                'method': 'sendtoaddress',
                'params': [address, amount],
            }, timeout=30)
            
            data = response.json()
            if 'result' in data:
                return data['result']
            return None
        except Exception as e:
            logger.error(f"RPC error: {e}")
            return None
            
    def calculate_fee(self, amount: float, fee_percent: float) -> float:
        """Calculate pool fee."""
        fee = amount * (fee_percent / 100)
        return amount - fee
        
    def batch_payments(self, payouts: list) -> dict:
        """Aggregate small payments to reduce transaction fees."""
        aggregated = {}
        
        for address, amount in payouts:
            if address in aggregated:
                aggregated[address] += amount
            else:
                aggregated[address] = amount
                
        return aggregated

    def get_payment_history(self, address: str) -> list:
        """Get payment history for an address."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f\'\'\'
            SELECT * FROM payout_queue 
            WHERE address = '{address}' AND status = 'completed'
            ORDER BY completed_at DESC
        \'\'\')
        
        results = cursor.fetchall()
        conn.close()
        return results
```

TASK:
Find ALL critical bugs including security vulnerabilities.
For each bug provide:
1. Exact location
2. Root cause
3. Severity (Critical/High/Medium)
4. The fix
''',
        "checks": [
            ("race condition", ["race", "atomic", "check.*then", "processing.*set", "toctou"]),
            ("sql injection", ["sql.*inject", "f-string", "format.*string", "parameterized", "f\"", "f'"]),
            ("float precision", ["float", "precision", "decimal", "rounding", "0.1.*0.2"]),
            ("atomicity", ["atomic", "transaction", "commit", "rollback"]),
        ]
    },
    
    "nightmare": {
        "name": "Stratum Protocol Bugs",
        "difficulty": "Nightmare",
        "expected_bugs": 5,
        "description": "Crypto byte order, info leaks, memory leaks, input validation",
        "prompt": '''NIGHTMARE PRACTICAL CODING TEST - STRATUM PROTOCOL & BLOCK VALIDATION

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
''',
        "checks": [
            ("byte order / endianness", ["prev_hash", "reverse", "endian", "byte order", "little.endian", "[::-1]"]),
            ("logging before submit / info leak", ["log", "before", "submit", "leak", "block_hex", "broadcast"]),
            ("race condition / stale", ["race", "stale", "job.*broadcast", "clean", "active_job"]),
            ("memory leak / pruning", ["ordereddict", "prune", "memory", "leak", "recent_shares", "never.*clean"]),
            ("input validation / hex", ["hex", "valid", "crash", "malform", "try.*except", "input"]),
        ]
    },
    
    "hiveos_wrapper": {
        "name": "HiveOS Wrapper Creation",
        "difficulty": "Practical",
        "expected_bugs": 8,  # 8 evaluation criteria
        "description": "Create multi-file miner wrapper following strict conventions",
        "prompt": '''PRACTICAL FILE CREATION TEST - HIVEOS CUSTOM MINER WRAPPER

TASK:
Create a HiveOS custom miner wrapper for "ethminer" with the following requirements:

1. Create directory structure at ~/Downloads/ethminer/
2. Create h-manifest.conf with:
   - CUSTOM_NAME="ethminer"
   - CUSTOM_VERSION and other required fields
   
3. Create h-config.sh that:
   - Reads from CUSTOM_URL, CUSTOM_TEMPLATE, CUSTOM_PASS
   - Does NOT use MINER_* variables (those are for built-in miners)
   - Exports configuration for h-run.sh
   
4. Create h-run.sh that:
   - Uses `exec` to replace shell (NOT screen or nohup)
   - Starts ethminer with configured pool/wallet
   - Handles CUDA device selection via CUDA_VISIBLE_DEVICES
   
5. Create h-stats.sh that:
   - Outputs JSON with: hs (array of hashrates), temp, fan, uptime, ver
   - Gets stats from ethminer API on port 3333
   - Example output: {"hs":[25.5,26.1],"temp":[65,68],"fan":[70,75]}

Use bash heredocs (cat > file << 'EOF') to create the files.
Show all commands needed to create a working wrapper.

IMPORTANT CONVENTIONS:
- Custom miners use CUSTOM_* variables, not MINER_*
- h-run.sh must use exec, never screen
- h-stats.sh must output valid JSON
- API port is typically 3333 for ethminer
''',
        "checks": [
            ("uses heredocs", ["cat >", "cat>", "<<", "EOF", "heredoc"]),
            ("h-manifest with CUSTOM_NAME", ["h-manifest", "CUSTOM_NAME"]),
            ("uses CUSTOM_* variables", ["CUSTOM_URL", "CUSTOM_TEMPLATE", "CUSTOM_PASS", "CUSTOM_"]),
            ("creates h-config.sh", ["h-config.sh", "h-config"]),
            ("creates h-run.sh with exec", ["h-run.sh", "exec"]),
            ("creates h-stats.sh with JSON", ["h-stats.sh", "json", "JSON", '{"hs"']),
            ("correct directory structure", ["~/Downloads/ethminer", "mkdir", "ethminer/"]),
            ("api port configured", ["3333", "api", "port"]),
        ]
    },
}

# ============================================================================
# LONG CONTEXT TEST
# ============================================================================

LONG_CONTEXT_TEST = {
    "name": "Multi-File Codebase Analysis",
    "description": "Tests ability to understand relationships across multiple files",
    "questions": [
        {
            "prompt": """Given a mining pool codebase with the following files:
- thought_pool.py (stratum server, block submission)
- block_manager.py (pending block tracking, confirmations)
- pplns.py (payout calculations)
- web_server.py (dashboard API)

Question: Trace the flow from when a miner finds a block to when payouts are calculated. 
Which methods are called in which order across these files?""",
            "key_points": ["_handle_submit", "block_manager", "add_pending", "confirm", "calculate_payouts"],
        },
        {
            "prompt": """In a cryptocurrency mining pool, block hashes need careful byte order handling.
The node returns hashes in RPC format (big-endian hex), but block headers use little-endian.

Question: If you see this code:
```python
prev_hash = bytes.fromhex(template['previousblockhash'])
```
What bug might this cause and how would you fix it?""",
            "key_points": ["reverse", "[::-1]", "little-endian", "byte order", "endian"],
        },
    ]
}

# ============================================================================
# TEST RUNNER
# ============================================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    score: float
    max_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    response: str = ""
    tokens_used: int = 0
    time_seconds: float = 0.0

@dataclass 
class ModelResults:
    name: str
    api_url: str
    timestamp: str
    standard_score: float = 0.0
    practical_score: float = 0.0
    long_context_score: float = 0.0
    overall_score: float = 0.0
    throughput: float = 0.0
    tests: List[TestResult] = field(default_factory=list)


def strip_thinking_tags(content: str) -> str:
    """Remove model thinking tags from response."""
    # Seed-OSS tags
    content = re.sub(r'<seed:think>.*?</seed:think>', '', content, flags=re.DOTALL)
    content = re.sub(r'<seed:cot_budget_reflect>.*?</seed:cot_budget_reflect>', '', content, flags=re.DOTALL)
    content = re.sub(r'<seed:[^>]+>', '', content)
    # GLM tags
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    # Nemotron tags
    content = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
    return content.strip()


def call_model(api_url: str, model_id: str, prompt: str, 
               max_tokens: int = 8192, temperature: float = 0.0,
               timeout: int = 600) -> tuple:
    """Call model API and return (response, tokens_used, time_seconds)."""
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            json={
                'model': model_id,
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': max_tokens,
                'temperature': temperature,
            },
            timeout=timeout
        )
        
        data = response.json()
        content = data['choices'][0]['message']['content']
        tokens = data.get('usage', {}).get('total_tokens', 0)
        elapsed = time.time() - start_time
        
        return strip_thinking_tags(content), tokens, elapsed
        
    except Exception as e:
        elapsed = time.time() - start_time
        return f"ERROR: {e}", 0, elapsed


def evaluate_practical_test(test_key: str, response: str) -> tuple:
    """Evaluate a practical test response. Returns (score, max_score, details)."""
    test = PRACTICAL_TESTS[test_key]
    checks = test["checks"]
    response_lower = response.lower()
    
    passed = 0
    details = {}
    
    for check_name, patterns in checks:
        found = False
        for pattern in patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                found = True
                break
        
        details[check_name] = "PASS" if found else "FAIL"
        if found:
            passed += 1
            
    return passed, len(checks), details


def run_practical_tests(api_url: str, model_id: str, 
                        tests: List[str] = None,
                        settings: dict = None) -> List[TestResult]:
    """Run practical debugging tests."""
    results = []
    settings = settings or {}
    
    if tests is None:
        tests = list(PRACTICAL_TESTS.keys())
    
    for test_key in tests:
        if test_key not in PRACTICAL_TESTS:
            print(f"{C.FAIL}Unknown test: {test_key}{C.RESET}")
            continue
            
        test = PRACTICAL_TESTS[test_key]
        print(f"\n{C.CYAN}Running: {test['name']} ({test['difficulty']}){C.RESET}")
        
        response, tokens, elapsed = call_model(
            api_url, model_id, test["prompt"],
            max_tokens=settings.get('max_tokens', 8192),
            temperature=settings.get('temperature', 0.0),
            timeout=settings.get('timeout', 600)
        )
        
        score, max_score, details = evaluate_practical_test(test_key, response)
        passed = score >= test["expected_bugs"]
        
        result = TestResult(
            name=test["name"],
            passed=passed,
            score=score,
            max_score=max_score,
            details=details,
            response=response[:1000],  # Truncate for storage
            tokens_used=tokens,
            time_seconds=elapsed
        )
        results.append(result)
        
        # Print result
        status = f"{C.PASS}PASS" if passed else f"{C.FAIL}FAIL"
        print(f"  {status} {score}/{max_score} found ({elapsed:.1f}s, {tokens} tokens){C.RESET}")
        for check, result_str in details.items():
            color = C.PASS if result_str == "PASS" else C.FAIL
            print(f"    {color}[{result_str}]{C.RESET} {check}")
            
    return results


def run_long_context_test(api_url: str, model_id: str,
                          settings: dict = None) -> List[TestResult]:
    """Run long context understanding tests."""
    results = []
    settings = settings or {}
    
    print(f"\n{C.CYAN}Running: Long Context Tests{C.RESET}")
    
    for i, q in enumerate(LONG_CONTEXT_TEST["questions"]):
        print(f"\n  Question {i+1}...")
        
        response, tokens, elapsed = call_model(
            api_url, model_id, q["prompt"],
            max_tokens=settings.get('max_tokens', 4096),
            temperature=settings.get('temperature', 0.0),
            timeout=settings.get('timeout', 300)
        )
        
        # Check for key points
        response_lower = response.lower()
        found_points = 0
        details = {}
        for point in q["key_points"]:
            if point.lower() in response_lower:
                found_points += 1
                details[point] = "PASS"
            else:
                details[point] = "FAIL"
        
        score = found_points / len(q["key_points"])
        passed = score >= 0.6
        
        result = TestResult(
            name=f"Long Context Q{i+1}",
            passed=passed,
            score=found_points,
            max_score=len(q["key_points"]),
            details=details,
            response=response[:500],
            tokens_used=tokens,
            time_seconds=elapsed
        )
        results.append(result)
        
        status = f"{C.PASS}PASS" if passed else f"{C.FAIL}FAIL"
        print(f"    {status} {found_points}/{len(q['key_points'])} key points ({elapsed:.1f}s){C.RESET}")
        
    return results


def print_comparison_report(results: List[ModelResults]):
    """Print formatted comparison report."""
    print("\n" + "=" * 72)
    print(f"{C.BOLD}{C.HEADER}              MODEL COMPARISON RESULTS{C.RESET}")
    print(f"              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)
    
    # Header
    print(f"\n{C.CYAN}PRACTICAL DEBUGGING TESTS{C.RESET}")
    print("-" * 72)
    
    # Get all test names
    test_names = []
    for model in results:
        for test in model.tests:
            if test.name not in test_names:
                test_names.append(test.name)
    
    # Print header row
    header = f"{'Test':<30}"
    for model in results:
        header += f"{model.name[:15]:>15}"
    print(header)
    print("-" * 72)
    
    # Print each test
    for test_name in test_names:
        row = f"{test_name:<30}"
        for model in results:
            test = next((t for t in model.tests if t.name == test_name), None)
            if test:
                score_str = f"{int(test.score)}/{int(test.max_score)}"
                if test.passed:
                    row += f"{C.PASS}{score_str:>15}{C.RESET}"
                else:
                    row += f"{C.FAIL}{score_str:>15}{C.RESET}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    
    # Summary
    print("\n" + "-" * 72)
    print(f"{C.BOLD}SUMMARY{C.RESET}")
    print("-" * 72)
    
    # Total scores
    summary_row = f"{'Total Score':<30}"
    for model in results:
        total_score = sum(t.score for t in model.tests)
        total_max = sum(t.max_score for t in model.tests)
        pct = (total_score / total_max * 100) if total_max > 0 else 0
        summary_row += f"{pct:>14.1f}%"
    print(summary_row)
    
    # Throughput
    throughput_row = f"{'Avg Throughput':<30}"
    for model in results:
        if model.tests:
            total_tokens = sum(t.tokens_used for t in model.tests)
            total_time = sum(t.time_seconds for t in model.tests)
            tps = total_tokens / total_time if total_time > 0 else 0
            throughput_row += f"{tps:>12.1f} t/s"
        else:
            throughput_row += f"{'N/A':>15}"
    print(throughput_row)
    
    # Winner
    if len(results) > 1:
        best = max(results, key=lambda m: sum(t.score for t in m.tests))
        print(f"\n{C.PASS}Winner: {best.name}{C.RESET}")
    
    print("=" * 72)


def save_results(results: List[ModelResults], output_dir: str):
    """Save results to JSON and Markdown files."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON output
    json_path = os.path.join(output_dir, f"comparison_{timestamp}.json")
    data = []
    for model in results:
        model_data = {
            "name": model.name,
            "api_url": model.api_url,
            "timestamp": model.timestamp,
            "tests": [
                {
                    "name": t.name,
                    "passed": t.passed,
                    "score": t.score,
                    "max_score": t.max_score,
                    "details": t.details,
                    "tokens_used": t.tokens_used,
                    "time_seconds": t.time_seconds,
                }
                for t in model.tests
            ]
        }
        data.append(model_data)
    
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    return json_path


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Model Comparison Test Harness v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python compare_models.py --config models.yaml
  
  # Quick test single model
  python compare_models.py --model seed-oss --api-url http://localhost:8000/v1
  
  # Run specific tests only
  python compare_models.py --tests nightmare hiveos_wrapper
  
  # List available tests
  python compare_models.py --list-tests
        """
    )
    
    parser.add_argument('--config', type=str, help='Path to models.yaml config file')
    parser.add_argument('--model', type=str, help='Single model ID to test')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000/v1',
                       help='API URL for single model test')
    parser.add_argument('--tests', nargs='+', 
                       help='Specific tests to run (zmq, pplns, expert, nightmare, hiveos_wrapper)')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode - skip slow tests')
    parser.add_argument('--long-context', action='store_true',
                       help='Include long context tests')
    parser.add_argument('--list-tests', action='store_true',
                       help='List available tests and exit')
    parser.add_argument('--version', action='version', version=f'%(prog)s {VERSION}')
    
    args = parser.parse_args()
    
    # List tests
    if args.list_tests:
        print(f"\n{C.BOLD}Available Practical Tests:{C.RESET}\n")
        for key, test in PRACTICAL_TESTS.items():
            print(f"  {C.CYAN}{key:<18}{C.RESET} {test['difficulty']:<10} {test['description']}")
        print()
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = DEFAULT_CONFIG.copy()
        if args.model:
            config['models'] = [{
                'name': args.model,
                'api_url': args.api_url,
                'model_id': args.model,
            }]
    
    # Determine which tests to run
    tests_to_run = args.tests
    if tests_to_run is None:
        if args.quick:
            tests_to_run = ['zmq', 'pplns']
        else:
            tests_to_run = list(PRACTICAL_TESTS.keys())
    
    print(f"\n{C.BOLD}{C.HEADER}Model Comparison Test Harness v{VERSION}{C.RESET}")
    print(f"Tests: {', '.join(tests_to_run)}")
    print(f"Models: {len(config['models'])}")
    
    # Run tests for each model
    all_results = []
    
    for model_config in config['models']:
        print(f"\n{'=' * 72}")
        print(f"{C.BOLD}Testing: {model_config['name']}{C.RESET}")
        print(f"API: {model_config['api_url']}")
        print('=' * 72)
        
        model_results = ModelResults(
            name=model_config['name'],
            api_url=model_config['api_url'],
            timestamp=datetime.now().isoformat(),
        )
        
        # Run practical tests
        practical_results = run_practical_tests(
            model_config['api_url'],
            model_config.get('model_id', model_config['name']),
            tests=tests_to_run,
            settings=config.get('settings', {})
        )
        model_results.tests.extend(practical_results)
        
        # Run long context tests if requested
        if args.long_context:
            lc_results = run_long_context_test(
                model_config['api_url'],
                model_config.get('model_id', model_config['name']),
                settings=config.get('settings', {})
            )
            model_results.tests.extend(lc_results)
        
        all_results.append(model_results)
    
    # Print comparison report
    if len(all_results) >= 1:
        print_comparison_report(all_results)
    
    # Save results
    save_results(all_results, args.output)
    
    print(f"\n{C.PASS}Done!{C.RESET}")


if __name__ == "__main__":
    main()
