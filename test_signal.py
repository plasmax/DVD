"""
Minimal test to debug Ctrl+C / signal handling.

Run these one at a time and press Ctrl+C during the "training" loop.
The test passes if you see the "SIGINT received" message and the
"save checkpoint?" prompt instead of an immediate exit.

Test 1 - direct:
    python test_signal.py

Test 2 - with tee:
    python test_signal.py 2>&1 | tee /tmp/signal_test.log

Test 3 - via accelerate launch (how you normally run):
    accelerate launch --config_file train_config/accelerate_config/accelerate_acc4.yaml test_signal.py

Test 4 - accelerate + tee:
    accelerate launch --config_file train_config/accelerate_config/accelerate_acc4.yaml test_signal.py 2>&1 | tee /tmp/signal_test.log

Test 5 - via bash wrapper + tee (closest to your real setup):
    bash -c 'accelerate launch --config_file train_config/accelerate_config/accelerate_acc4.yaml test_signal.py' 2>&1 | tee /tmp/signal_test.log

Test 6 - direct python + tee:
    python test_signal.py 2>&1 | tee /tmp/signal_test.log
"""
import os
import signal
import sys
import time

_interrupt_requested = False
_interrupt_count = 0


def _sigint_handler(signum, frame):
    del frame
    global _interrupt_requested, _interrupt_count
    _interrupt_count += 1
    if _interrupt_count >= 2:
        raise KeyboardInterrupt
    _interrupt_requested = True
    sig_name = signal.Signals(signum).name
    try:
        print(
            f"\n[HANDLER] {sig_name} received. Will pause at next safe point."
            " Press Ctrl+C again to exit immediately.",
            file=sys.stderr,
            flush=True,
        )
    except (BrokenPipeError, OSError):
        pass


def fake_save_prompt():
    print("\n[SAVE] Would you like to save a checkpoint? [y/N]: ", end="", flush=True)
    if sys.stdin.isatty():
        answer = input().strip().lower()
        if answer in ("y", "yes"):
            print("[SAVE] Saving... (pretend)", flush=True)
        else:
            print("[SAVE] Skipping save.", flush=True)
    else:
        print("\n[SAVE] No TTY, would auto-save.", flush=True)


def main():
    global _interrupt_requested, _interrupt_count

    print(f"PID: {os.getpid()}", flush=True)
    print(f"PPID: {os.getppid()}", flush=True)
    print(f"Process group: {os.getpgrp()}", flush=True)
    print(f"stdin is TTY: {sys.stdin.isatty()}", flush=True)
    print(f"stdout is TTY: {sys.stdout.isatty()}", flush=True)
    print(f"stderr is TTY: {sys.stderr.isatty()}", flush=True)

    # Register handlers
    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    current_handler = signal.getsignal(signal.SIGINT)
    print(f"SIGINT handler after registration: {current_handler}", flush=True)

    print("\n--- Simulated training loop (press Ctrl+C) ---", flush=True)

    try:
        for step in range(1, 10001):
            # Check handler is still ours every 10 steps
            if step % 10 == 0:
                h = signal.getsignal(signal.SIGINT)
                if h is not _sigint_handler:
                    print(
                        f"[WARNING] SIGINT handler was overridden at step {step}! "
                        f"Now: {h}",
                        file=sys.stderr,
                        flush=True,
                    )
                    signal.signal(signal.SIGINT, _sigint_handler)

            # Simulate work (~1 second per "step")
            print(f"Step {step}...", flush=True)
            time.sleep(1)

            # Check interrupt flag (like maybe_save_and_exit_on_interrupt)
            if _interrupt_requested:
                print(f"\n[LOOP] Interrupt detected at step {step}.", flush=True)
                _interrupt_count = 0
                fake_save_prompt()
                _interrupt_requested = False
                print("[LOOP] Exiting cleanly.", flush=True)
                return

    except KeyboardInterrupt:
        print(
            "\n[EXCEPT] KeyboardInterrupt caught in outer try/except.",
            flush=True,
        )
        fake_save_prompt()

    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()
