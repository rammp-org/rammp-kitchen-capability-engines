"""A terminal stand-in for the touchscreen: run one of button_press's UI actions with its real panels.

It plays two parts the real stack plays on the robot:

- the UI bridge (rammp_ui_bridge): serves /<bridge>/request_user_input and its /update service, prints every panel
  the engine sends (title, message, progress, buttons) and sends back the button you type;
- the orchestrator's goal: sends /<engine>/execute_action with a session token, so the engine's
  ctx.update_user_panel() talks to this bridge instead of skipping (it does nothing without a token).

The node, arm, cameras and planner are the real ones: with the node's allow_execute:=true the arm MOVES. Someone on
the e-stop. The token is made up here and the arm driver gets it too: that is harmless only with the driver's
arbitration_mode disabled (the december_2026 manifest's setting); under "enforced" its goals are refused.

Don't run it while the real UI stack (scripts/kitchen-ui/up.sh, ui_bridge_node) is up: two bridges of one name.

Run in the capability's image (it has rclpy and the interfaces), on the robot's DDS domain:

    python3 cli_ui.py                       # heat_food on button_press
    python3 cli_ui.py --action heat_food --engine button_press --bridge ui_bridge_node

Type a button's number or id at the prompt; "stop" works on every panel (as on the touchscreen). Ctrl-C cancels the
action as the orchestrator would.
"""

import argparse
import json
import os
import sys
import threading
import time

import rclpy
from rammp_orchestrator_interfaces.action import ExecuteAction
from rammp_ui_bridge_interfaces.action import RequestUserInput
from rammp_ui_bridge_interfaces.srv import UpdateUserInputPanel
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node


def show(panel):
    """Print one panel: its title, message, progress and buttons."""
    out = [f"\n=== {panel.get('title', '')} ==="]
    if panel.get("message"):
        out.append(panel["message"])
    pr = panel.get("progress")
    if pr:
        out.append(f"[{pr.get('phase', '')} {float(pr.get('pct', 0)):.0f}%]")
    buttons = panel.get("buttons") or []
    if buttons:
        out.append("  ".join(f"[{i}] {b.get('label', b.get('button_id'))}" for i, b in enumerate(buttons, 1)))
    print("\n".join(out), flush=True)


class CliBridge(Node):
    """The UI bridge's two interfaces, one open panel at a time, answered from the terminal."""

    def __init__(self, bridge_name):
        super().__init__(bridge_name)
        group = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.panel = None  # the open goal's panel (dict) or None
        self.shown = None  # the last panel printed (to skip identical refreshes)
        self.answer = None  # the button id typed for the open goal
        self.token = None
        self.open_goal = None
        self.server = ActionServer(
            self,
            RequestUserInput,
            "~/request_user_input",
            execute_callback=self._execute,
            goal_callback=self._goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=group,
        )
        self.create_service(UpdateUserInputPanel, "~/request_user_input/update", self._update, callback_group=group)

    def _goal(self, goal):
        with self.lock:
            if self.open_goal is not None:  # the real bridge allows one open goal per token
                return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _set_panel(self, request_json):
        payload = json.loads(request_json or "{}")
        if "panel_json" not in payload:
            return
        panel = json.loads(payload["panel_json"])
        with self.lock:
            self.panel = panel
            if panel == self.shown:
                return
            self.shown = panel
        show(panel)

    def _execute(self, goal_handle):
        with self.lock:
            self.open_goal, self.answer, self.token = goal_handle, None, bytes(goal_handle.request.token)
        self._set_panel(goal_handle.request.request_json)
        result = RequestUserInput.Result()
        try:
            while rclpy.ok():
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.result_json = json.dumps({"status": "cancelled"})
                    return result
                with self.lock:
                    answer = self.answer
                if answer is not None:
                    goal_handle.succeed()
                    result.result_json = json.dumps({"status": "button", "button_id": answer})
                    return result
                time.sleep(0.05)
            goal_handle.abort()
            result.result_json = json.dumps({"status": "aborted", "reason": "bridge shutting down"})
            return result
        finally:
            with self.lock:
                self.open_goal = self.panel = None

    def _update(self, request, response):
        with self.lock:
            ok = self.open_goal is not None and bytes(request.token) == self.token
        response.accepted = ok
        response.message = "" if ok else "no open goal for this token"
        if ok:
            self._set_panel(request.request_json)
        return response

    def press(self, typed):
        """A typed answer: a button's number or id on the open panel. Returns why it wasn't taken, or ""."""
        with self.lock:
            if self.panel is None:
                return "no panel is open"
            buttons = self.panel.get("buttons") or []
            ids = [b.get("button_id") for b in buttons]
            if typed.isdigit() and 1 <= int(typed) <= len(ids):
                typed = ids[int(typed) - 1]
            if typed not in ids:
                return f"not a button here: {', '.join(ids) or 'none'}"
            self.answer = typed
            self.shown = None  # print the next panel even if it repeats
        return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--engine", default="button_press")
    ap.add_argument("--action", default="heat_food")
    ap.add_argument("--params", default="{}", help="the goal's params_json")
    ap.add_argument("--bridge", default="ui_bridge_node", help="the engine's ui_bridge_node_name")
    args = ap.parse_args()

    rclpy.init()
    bridge = CliBridge(args.bridge)
    executor = MultiThreadedExecutor()
    executor.add_node(bridge)
    threading.Thread(target=executor.spin, daemon=True).start()

    client = ActionClient(bridge, ExecuteAction, f"/{args.engine}/execute_action")
    print(f"waiting for /{args.engine}/execute_action ...", flush=True)
    if not client.wait_for_server(timeout_sec=30.0):
        sys.exit(f"no /{args.engine}/execute_action: is the {args.engine} node running?")
    token = os.urandom(16)
    goal = ExecuteAction.Goal(engine_id=args.engine, action_id=args.action, params_json=args.params, token=list(token))
    print(f"sending {args.action} to {args.engine} (allow_execute decides whether the arm moves)", flush=True)
    future = client.send_goal_async(goal)
    while not future.done():
        time.sleep(0.05)
    handle = future.result()
    if not handle.accepted:
        sys.exit("the engine refused the goal")
    result_future = handle.get_result_async()

    def read_keys():
        while not result_future.done():
            try:
                typed = input().strip()
            except EOFError:
                return
            if typed:
                why = bridge.press(typed)
                if why:
                    print(f"  ({why})", flush=True)

    threading.Thread(target=read_keys, daemon=True).start()
    try:
        while not result_future.done():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\ncancelling the action (as the orchestrator would) ...", flush=True)
        handle.cancel_goal_async()
        while not result_future.done():
            time.sleep(0.1)
    r = result_future.result().result.result
    status = {0: "success", 1: "failed", 2: "cancelled"}.get(r.status, r.status)
    print(f"\n=== action ended: {status} {r.error_code} ===\n{r.payload_json}", flush=True)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
