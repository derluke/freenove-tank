/**
 * Blockly LiveView hook — initialises the workspace with robot-specific blocks.
 *
 * Custom blocks:
 *   - drive_forward / drive_backward (with speed + duration)
 *   - turn_left / turn_right
 *   - stop
 *   - wait (seconds)
 *   - set_led (color picker)
 *   - led_off
 *   - move_servo (channel + angle)
 *
 * "Run" generates a JSON command list and pushes it to the LiveView.
 */

// We load Blockly from CDN in root.html.heex
const BLOCKLY_CDN = "https://unpkg.com/blockly/blockly_compressed.js";

function defineRobotBlocks(Blockly) {
  // --- Drive Forward ---
  Blockly.Blocks["drive_forward"] = {
    init: function () {
      this.appendDummyInput()
        .appendField("Drive forward at speed")
        .appendField(new Blockly.FieldNumber(2000, 0, 4095), "SPEED")
        .appendField("for")
        .appendField(new Blockly.FieldNumber(1, 0.1, 30), "DURATION")
        .appendField("seconds");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(230);
      this.setTooltip("Drive both treads forward");
    },
  };

  // --- Drive Backward ---
  Blockly.Blocks["drive_backward"] = {
    init: function () {
      this.appendDummyInput()
        .appendField("Drive backward at speed")
        .appendField(new Blockly.FieldNumber(2000, 0, 4095), "SPEED")
        .appendField("for")
        .appendField(new Blockly.FieldNumber(1, 0.1, 30), "DURATION")
        .appendField("seconds");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(230);
    },
  };

  // --- Turn Left ---
  Blockly.Blocks["turn_left"] = {
    init: function () {
      this.appendDummyInput()
        .appendField("Turn left at speed")
        .appendField(new Blockly.FieldNumber(1500, 0, 4095), "SPEED")
        .appendField("for")
        .appendField(new Blockly.FieldNumber(0.5, 0.1, 30), "DURATION")
        .appendField("seconds");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(120);
    },
  };

  // --- Turn Right ---
  Blockly.Blocks["turn_right"] = {
    init: function () {
      this.appendDummyInput()
        .appendField("Turn right at speed")
        .appendField(new Blockly.FieldNumber(1500, 0, 4095), "SPEED")
        .appendField("for")
        .appendField(new Blockly.FieldNumber(0.5, 0.1, 30), "DURATION")
        .appendField("seconds");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(120);
    },
  };

  // --- Stop ---
  Blockly.Blocks["robot_stop"] = {
    init: function () {
      this.appendDummyInput().appendField("Stop");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(0);
    },
  };

  // --- Wait ---
  Blockly.Blocks["wait_seconds"] = {
    init: function () {
      this.appendDummyInput()
        .appendField("Wait")
        .appendField(new Blockly.FieldNumber(1, 0.1, 60), "SECONDS")
        .appendField("seconds");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(60);
    },
  };

  // --- Set LED Color ---
  Blockly.Blocks["set_led"] = {
    init: function () {
      this.appendDummyInput()
        .appendField("Set LEDs to")
        .appendField(new Blockly.FieldColour("#ff0000"), "COLOUR");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(300);
    },
  };

  // --- LED Off ---
  Blockly.Blocks["led_off"] = {
    init: function () {
      this.appendDummyInput().appendField("Turn LEDs off");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(300);
    },
  };

  // --- Servo ---
  Blockly.Blocks["move_servo"] = {
    init: function () {
      this.appendDummyInput()
        .appendField("Move servo")
        .appendField(
          new Blockly.FieldDropdown([
            ["Head tilt", "0"],
            ["Head pan", "1"],
            ["Arm", "2"],
          ]),
          "CHANNEL"
        )
        .appendField("to angle")
        .appendField(new Blockly.FieldAngle(90), "ANGLE");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(180);
    },
  };
}

function blockToCommand(block) {
  switch (block.type) {
    case "drive_forward": {
      const speed = block.getFieldValue("SPEED");
      const dur = block.getFieldValue("DURATION");
      return { type: "drive", left: speed, right: speed, duration: dur };
    }
    case "drive_backward": {
      const speed = block.getFieldValue("SPEED");
      const dur = block.getFieldValue("DURATION");
      return { type: "drive", left: -speed, right: -speed, duration: dur };
    }
    case "turn_left": {
      const speed = block.getFieldValue("SPEED");
      const dur = block.getFieldValue("DURATION");
      return { type: "drive", left: -speed, right: speed, duration: dur };
    }
    case "turn_right": {
      const speed = block.getFieldValue("SPEED");
      const dur = block.getFieldValue("DURATION");
      return { type: "drive", left: speed, right: -speed, duration: dur };
    }
    case "robot_stop":
      return { type: "stop" };
    case "wait_seconds":
      return { type: "wait", seconds: block.getFieldValue("SECONDS") };
    case "set_led": {
      const hex = block.getFieldValue("COLOUR");
      const r = parseInt(hex.slice(1, 3), 16);
      const g = parseInt(hex.slice(3, 5), 16);
      const b = parseInt(hex.slice(5, 7), 16);
      return { type: "led", r, g, b };
    }
    case "led_off":
      return { type: "led_off" };
    case "move_servo":
      return {
        type: "servo",
        channel: parseInt(block.getFieldValue("CHANNEL")),
        angle: block.getFieldValue("ANGLE"),
      };
    default:
      return null;
  }
}

function workspaceToCommands(workspace) {
  const topBlocks = workspace.getTopBlocks(true);
  const commands = [];
  for (const top of topBlocks) {
    let block = top;
    while (block) {
      const cmd = blockToCommand(block);
      if (cmd) commands.push(cmd);
      block = block.getNextBlock();
    }
  }
  return commands;
}

const TOOLBOX = {
  kind: "categoryToolbox",
  contents: [
    {
      kind: "category",
      name: "🚗 Movement",
      colour: "230",
      contents: [
        { kind: "block", type: "drive_forward" },
        { kind: "block", type: "drive_backward" },
        { kind: "block", type: "turn_left" },
        { kind: "block", type: "turn_right" },
        { kind: "block", type: "robot_stop" },
      ],
    },
    {
      kind: "category",
      name: "⏱ Timing",
      colour: "60",
      contents: [{ kind: "block", type: "wait_seconds" }],
    },
    {
      kind: "category",
      name: "💡 LEDs",
      colour: "300",
      contents: [
        { kind: "block", type: "set_led" },
        { kind: "block", type: "led_off" },
      ],
    },
    {
      kind: "category",
      name: "🦾 Servo",
      colour: "180",
      contents: [{ kind: "block", type: "move_servo" }],
    },
  ],
};

// --- Hook: initialise Blockly workspace ---
export const Blockly = {
  mounted() {
    const self = this;
    this.loadBlockly().then((BlocklyLib) => {
      defineRobotBlocks(BlocklyLib);
      self.workspace = BlocklyLib.inject(self.el, {
        toolbox: TOOLBOX,
        grid: { spacing: 20, length: 3, colour: "#444", snap: true },
        zoom: { controls: true, wheel: true, startScale: 1.0 },
        trashcan: true,
        theme: BlocklyLib.Themes.Dark || undefined,
      });
      // Expose for RunBlockly hook
      window.__blocklyWorkspace = self.workspace;
    });
  },

  async loadBlockly() {
    if (window.Blockly) return window.Blockly;
    return new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = BLOCKLY_CDN;
      script.onload = () => resolve(window.Blockly);
      script.onerror = reject;
      document.head.appendChild(script);
    });
  },

  destroyed() {
    if (this.workspace) {
      this.workspace.dispose();
      window.__blocklyWorkspace = null;
    }
  },
};

// --- Hook: Run button sends commands to LiveView ---
export const RunBlockly = {
  mounted() {
    this.el.addEventListener("click", () => {
      const workspace = window.__blocklyWorkspace;
      if (!workspace) return;
      const commands = workspaceToCommands(workspace);
      this.pushEvent("run_program", { commands });
    });
  },
};
