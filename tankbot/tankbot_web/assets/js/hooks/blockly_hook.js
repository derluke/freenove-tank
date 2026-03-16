/**
 * Blockly LiveView hook — visual programming for the tank robot.
 *
 * Uses the blockly npm package with the JavaScript code generator.
 * Built-in blocks (loops, if/else, variables, math, logic, text) work
 * out of the box. Custom robot blocks generate async JS that calls
 * into a `robot` API object which forwards commands to the LiveView.
 */

import * as BlocklyLib from "blockly/core";
import "blockly/blocks";
import * as En from "blockly/msg/en";
import { javascriptGenerator, Order } from "blockly/javascript";

BlocklyLib.setLocale(En);

const LED_COLORS = [
  ["Red", "ff0000"],
  ["Green", "00ff00"],
  ["Blue", "0000ff"],
  ["Yellow", "ffff00"],
  ["Purple", "ff00ff"],
  ["Cyan", "00ffff"],
  ["White", "ffffff"],
  ["Orange", "ff8800"],
  ["Pink", "ff66cc"],
];

// ---------------------------------------------------------------------------
// Block definitions
// ---------------------------------------------------------------------------

function defineRobotBlocks() {
  // --- Start block (hat block — no previous statement) ---
  BlocklyLib.Blocks["program_start"] = {
    init() {
      this.appendDummyInput().appendField("When program starts");
      this.setNextStatement(true, null);
      this.setColour(65);
      this.setTooltip("The program begins here");
      this.setDeletable(false);
    },
  };

  BlocklyLib.Blocks["drive_forward"] = {
    init() {
      this.appendDummyInput()
        .appendField("Drive forward at speed")
        .appendField(new BlocklyLib.FieldNumber(2000, 0, 4095), "SPEED")
        .appendField("for")
        .appendField(new BlocklyLib.FieldNumber(1, 0.1, 30), "DURATION")
        .appendField("seconds");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(230);
      this.setTooltip("Drive both treads forward");
    },
  };

  BlocklyLib.Blocks["drive_backward"] = {
    init() {
      this.appendDummyInput()
        .appendField("Drive backward at speed")
        .appendField(new BlocklyLib.FieldNumber(2000, 0, 4095), "SPEED")
        .appendField("for")
        .appendField(new BlocklyLib.FieldNumber(1, 0.1, 30), "DURATION")
        .appendField("seconds");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(230);
    },
  };

  BlocklyLib.Blocks["turn_left"] = {
    init() {
      this.appendDummyInput()
        .appendField("Turn left at speed")
        .appendField(new BlocklyLib.FieldNumber(1500, 0, 4095), "SPEED")
        .appendField("for")
        .appendField(new BlocklyLib.FieldNumber(0.5, 0.1, 30), "DURATION")
        .appendField("seconds");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(120);
    },
  };

  BlocklyLib.Blocks["turn_right"] = {
    init() {
      this.appendDummyInput()
        .appendField("Turn right at speed")
        .appendField(new BlocklyLib.FieldNumber(1500, 0, 4095), "SPEED")
        .appendField("for")
        .appendField(new BlocklyLib.FieldNumber(0.5, 0.1, 30), "DURATION")
        .appendField("seconds");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(120);
    },
  };

  BlocklyLib.Blocks["robot_stop"] = {
    init() {
      this.appendDummyInput().appendField("Stop");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(0);
    },
  };

  BlocklyLib.Blocks["wait_seconds"] = {
    init() {
      this.appendDummyInput()
        .appendField("Wait")
        .appendField(new BlocklyLib.FieldNumber(1, 0.1, 60), "SECONDS")
        .appendField("seconds");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(60);
    },
  };

  BlocklyLib.Blocks["set_led"] = {
    init() {
      this.appendDummyInput()
        .appendField("Set LEDs to")
        .appendField(
          new BlocklyLib.FieldDropdown(LED_COLORS.map(([name, hex]) => [name, hex])),
          "COLOUR"
        );
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(300);
    },
  };

  BlocklyLib.Blocks["led_off"] = {
    init() {
      this.appendDummyInput().appendField("Turn LEDs off");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(300);
    },
  };

  BlocklyLib.Blocks["move_arm"] = {
    init() {
      this.appendDummyInput()
        .appendField("Move arm to")
        .appendField(new BlocklyLib.FieldNumber(90, 75, 150), "ANGLE")
        .appendField("degrees");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(180);
    },
  };

  BlocklyLib.Blocks["move_grabber"] = {
    init() {
      this.appendDummyInput()
        .appendField("Grabber")
        .appendField(
          new BlocklyLib.FieldDropdown([
            ["Open", "90"],
            ["Close", "140"],
            ["Half open", "115"],
          ]),
          "ANGLE"
        );
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(180);
    },
  };

  BlocklyLib.Blocks["get_distance"] = {
    init() {
      this.appendDummyInput().appendField("distance (cm)");
      this.setOutput(true, "Number");
      this.setColour(20);
      this.setTooltip("Current ultrasonic distance in cm");
    },
  };

  BlocklyLib.Blocks["log_message"] = {
    init() {
      this.appendValueInput("MSG").appendField("Log");
      this.setPreviousStatement(true, null);
      this.setNextStatement(true, null);
      this.setColour(160);
    },
  };
}

// ---------------------------------------------------------------------------
// JavaScript code generators for custom blocks
// ---------------------------------------------------------------------------

function defineGenerators() {
  javascriptGenerator.forBlock["program_start"] = function () {
    return "";
  };

  javascriptGenerator.forBlock["drive_forward"] = function (block) {
    const speed = block.getFieldValue("SPEED");
    const dur = block.getFieldValue("DURATION");
    return `await robot.drive(${speed}, ${speed}, ${dur});\n`;
  };

  javascriptGenerator.forBlock["drive_backward"] = function (block) {
    const speed = block.getFieldValue("SPEED");
    const dur = block.getFieldValue("DURATION");
    return `await robot.drive(${-speed}, ${-speed}, ${dur});\n`;
  };

  javascriptGenerator.forBlock["turn_left"] = function (block) {
    const speed = block.getFieldValue("SPEED");
    const dur = block.getFieldValue("DURATION");
    return `await robot.drive(${-speed}, ${speed}, ${dur});\n`;
  };

  javascriptGenerator.forBlock["turn_right"] = function (block) {
    const speed = block.getFieldValue("SPEED");
    const dur = block.getFieldValue("DURATION");
    return `await robot.drive(${speed}, ${-speed}, ${dur});\n`;
  };

  javascriptGenerator.forBlock["robot_stop"] = function () {
    return `await robot.stop();\n`;
  };

  javascriptGenerator.forBlock["wait_seconds"] = function (block) {
    const secs = block.getFieldValue("SECONDS");
    return `await robot.wait(${secs});\n`;
  };

  javascriptGenerator.forBlock["set_led"] = function (block) {
    const hex = block.getFieldValue("COLOUR");
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `await robot.led(${r}, ${g}, ${b});\n`;
  };

  javascriptGenerator.forBlock["led_off"] = function () {
    return `await robot.ledOff();\n`;
  };

  javascriptGenerator.forBlock["move_arm"] = function (block) {
    const angle = block.getFieldValue("ANGLE");
    return `await robot.arm(${angle});\n`;
  };

  javascriptGenerator.forBlock["move_grabber"] = function (block) {
    const angle = block.getFieldValue("ANGLE");
    return `await robot.grabber(${parseInt(angle)});\n`;
  };

  javascriptGenerator.forBlock["get_distance"] = function () {
    return [`robot.distance`, Order.ATOMIC];
  };

  javascriptGenerator.forBlock["log_message"] = function (block) {
    const msg = javascriptGenerator.valueToCode(block, "MSG", Order.NONE) || "''";
    return `await robot.log(${msg});\n`;
  };
}

// ---------------------------------------------------------------------------
// Toolbox
// ---------------------------------------------------------------------------

const TOOLBOX = {
  kind: "categoryToolbox",
  contents: [
    {
      kind: "category",
      name: "Movement",
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
      name: "Arm & Grabber",
      colour: "180",
      contents: [
        { kind: "block", type: "move_arm" },
        { kind: "block", type: "move_grabber" },
      ],
    },
    {
      kind: "category",
      name: "LEDs",
      colour: "300",
      contents: [
        { kind: "block", type: "set_led" },
        { kind: "block", type: "led_off" },
      ],
    },
    {
      kind: "category",
      name: "Sensors",
      colour: "20",
      contents: [{ kind: "block", type: "get_distance" }],
    },
    {
      kind: "category",
      name: "Timing",
      colour: "60",
      contents: [{ kind: "block", type: "wait_seconds" }],
    },
    { kind: "sep" },
    {
      kind: "category",
      name: "Logic",
      colour: "210",
      contents: [
        { kind: "block", type: "controls_if" },
        { kind: "block", type: "controls_ifelse" },
        { kind: "block", type: "logic_compare" },
        { kind: "block", type: "logic_operation" },
        { kind: "block", type: "logic_negate" },
        { kind: "block", type: "logic_boolean" },
      ],
    },
    {
      kind: "category",
      name: "Loops",
      colour: "120",
      contents: [
        {
          kind: "block",
          type: "controls_repeat_ext",
          inputs: {
            TIMES: {
              shadow: { type: "math_number", fields: { NUM: 3 } },
            },
          },
        },
        { kind: "block", type: "controls_whileUntil" },
        {
          kind: "block",
          type: "controls_for",
          fields: { VAR: "i" },
          inputs: {
            FROM: { shadow: { type: "math_number", fields: { NUM: 1 } } },
            TO: { shadow: { type: "math_number", fields: { NUM: 10 } } },
            BY: { shadow: { type: "math_number", fields: { NUM: 1 } } },
          },
        },
      ],
    },
    {
      kind: "category",
      name: "Math",
      colour: "230",
      contents: [
        { kind: "block", type: "math_number" },
        { kind: "block", type: "math_arithmetic" },
        {
          kind: "block",
          type: "math_random_int",
          inputs: {
            FROM: { shadow: { type: "math_number", fields: { NUM: 1 } } },
            TO: { shadow: { type: "math_number", fields: { NUM: 100 } } },
          },
        },
        { kind: "block", type: "math_modulo" },
      ],
    },
    {
      kind: "category",
      name: "Variables",
      colour: "330",
      custom: "VARIABLE",
    },
    {
      kind: "category",
      name: "Text",
      colour: "160",
      contents: [
        { kind: "block", type: "text" },
        { kind: "block", type: "text_join" },
        { kind: "block", type: "log_message" },
      ],
    },
  ],
};

// ---------------------------------------------------------------------------
// Robot API — bridge between generated code and LiveView
// ---------------------------------------------------------------------------

function createRobotAPI(hook) {
  let _stopped = false;

  function checkStopped() {
    if (_stopped) throw new Error("__PROGRAM_STOPPED__");
  }

  return {
    get distance() {
      return window.__robotDistance || 0;
    },

    async drive(left, right, duration) {
      checkStopped();
      hook.pushEvent("motor", { left, right });
      if (duration > 0) {
        await this.wait(duration);
        hook.pushEvent("stop", {});
      }
    },

    async stop() {
      hook.pushEvent("stop", {});
    },

    async wait(seconds) {
      checkStopped();
      const ms = seconds * 1000;
      const step = 50;
      for (let elapsed = 0; elapsed < ms; elapsed += step) {
        if (_stopped) throw new Error("__PROGRAM_STOPPED__");
        await new Promise((r) => setTimeout(r, Math.min(step, ms - elapsed)));
      }
    },

    async led(r, g, b) {
      checkStopped();
      hook.pushEvent("led", { r, g, b });
    },

    async ledOff() {
      checkStopped();
      hook.pushEvent("led_off", {});
    },

    async arm(angle) {
      checkStopped();
      hook.pushEvent("servo", { channel: 1, angle });
    },

    async grabber(angle) {
      checkStopped();
      hook.pushEvent("servo", { channel: 0, angle });
    },

    async log(msg) {
      checkStopped();
      appendLog(String(msg));
    },

    _stop() {
      _stopped = true;
    },
  };
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------

// Migrate saved workspace state to handle changed dropdown values
// (e.g. grabber "Close" was "150", now "140")
const FIELD_MIGRATIONS = {
  move_grabber: { ANGLE: { "150": "140" } },
};

function migrateState(state) {
  if (!state || !state.blocks || !state.blocks.blocks) return;
  const stack = [...state.blocks.blocks];
  while (stack.length > 0) {
    const block = stack.pop();
    const migrations = FIELD_MIGRATIONS[block.type];
    if (migrations && block.fields) {
      for (const [fieldName, mapping] of Object.entries(migrations)) {
        if (fieldName in block.fields && block.fields[fieldName] in mapping) {
          block.fields[fieldName] = mapping[block.fields[fieldName]];
        }
      }
    }
    if (block.next && block.next.block) stack.push(block.next.block);
    if (block.inputs) {
      for (const input of Object.values(block.inputs)) {
        if (input.block) stack.push(input.block);
        if (input.shadow) stack.push(input.shadow);
      }
    }
  }
}

let _blocksReady = false;

export const Blockly = {
  mounted() {
    const target = this.el.querySelector("#blockly-target") || this.el;

    if (!_blocksReady) {
      defineRobotBlocks();
      defineGenerators();
      _blocksReady = true;
    }

    const darkTheme = BlocklyLib.Theme.defineTheme("tankbot_dark", {
      base: BlocklyLib.Themes.Classic,
      componentStyles: {
        workspaceBackgroundColour: "#1f2937",
        toolboxBackgroundColour: "#111827",
        toolboxForegroundColour: "#e5e7eb",
        flyoutBackgroundColour: "#1f2937",
        flyoutForegroundColour: "#e5e7eb",
        flyoutOpacity: 0.95,
        scrollbarColour: "#4b5563",
        scrollbarOpacity: 0.6,
        insertionMarkerColour: "#60a5fa",
      },
      fontStyle: {
        family: "system-ui, sans-serif",
        size: 12,
      },
    });

    this.workspace = BlocklyLib.inject(target, {
      toolbox: TOOLBOX,
      grid: { spacing: 20, length: 3, colour: "#374151", snap: true },
      zoom: { controls: true, wheel: true, startScale: 1.0 },
      trashcan: true,
      theme: darkTheme,
    });

    window.__blocklyWorkspace = this.workspace;

    // Start block is placed by _setupProgramManager (via autosave restore or fresh)

    // Listen for distance updates from server (JS event, no DOM patch)
    this.handleEvent("distance_update", ({ distance }) => {
      window.__robotDistance = distance;
      const el = document.getElementById("blocks-distance");
      if (el) el.textContent = `Distance: ${distance.toFixed(1)} cm`;
    });

    // Listen for stop signal from server
    this.handleEvent("stop_program", () => {
      if (window.__currentRobot) {
        window.__currentRobot._stop();
      }
    });

    // --- Save / Load / Delete programs ---
    this._setupProgramManager();
  },

  _setupProgramManager() {
    const select = document.getElementById("program-select");
    const saveBtn = document.getElementById("save-btn");
    const loadBtn = document.getElementById("load-btn");
    const deleteBtn = document.getElementById("delete-btn");
    if (!select || !saveBtn || !loadBtn || !deleteBtn) return;

    const STORAGE_KEY = "tankbot_programs";
    const self = this;

    function getPrograms() {
      try {
        return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
      } catch { return {}; }
    }

    function savePrograms(programs) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(programs));
    }

    function refreshSelect() {
      const programs = getPrograms();
      const names = Object.keys(programs).sort();
      select.innerHTML = '<option value="">-- Programs --</option>';
      for (const name of names) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      }
    }

    saveBtn.addEventListener("click", () => {
      const name = prompt("Program name:");
      if (!name || !name.trim()) return;
      const ws = self.workspace;
      if (!ws) return;
      const state = BlocklyLib.serialization.workspaces.save(ws);
      const programs = getPrograms();
      programs[name.trim()] = state;
      savePrograms(programs);
      refreshSelect();
      select.value = name.trim();
      appendLog(`Saved: ${name.trim()}`);
    });

    loadBtn.addEventListener("click", () => {
      const name = select.value;
      if (!name) return;
      const programs = getPrograms();
      const state = programs[name];
      if (!state) return;
      const ws = self.workspace;
      if (!ws) return;
      migrateState(state);
      ws.clear();
      BlocklyLib.serialization.workspaces.load(state, ws);
      // Ensure there's always a start block
      const startBlocks = ws.getBlocksByType("program_start");
      if (startBlocks.length === 0) {
        const sb = ws.newBlock("program_start");
        sb.initSvg();
        sb.render();
        sb.moveBy(30, 30);
      }
      appendLog(`Loaded: ${name}`);
    });

    deleteBtn.addEventListener("click", () => {
      const name = select.value;
      if (!name) return;
      if (!confirm(`Delete program "${name}"?`)) return;
      const programs = getPrograms();
      delete programs[name];
      savePrograms(programs);
      refreshSelect();
      appendLog(`Deleted: ${name}`);
    });

    // Auto-save current workspace periodically
    let autoSaveTimer = null;
    if (this.workspace) {
      this.workspace.addChangeListener(() => {
        clearTimeout(autoSaveTimer);
        autoSaveTimer = setTimeout(() => {
          try {
            const state = BlocklyLib.serialization.workspaces.save(this.workspace);
            localStorage.setItem("tankbot_autosave", JSON.stringify(state));
          } catch {}
        }, 2000);
      });

      // Restore autosave on mount, or place a fresh start block
      let restored = false;
      try {
        const autosave = JSON.parse(localStorage.getItem("tankbot_autosave"));
        if (autosave) {
          migrateState(autosave);
          this.workspace.clear();
          BlocklyLib.serialization.workspaces.load(autosave, this.workspace);
          restored = true;
        }
      } catch {}

      // Ensure there's always a start block
      if (this.workspace.getBlocksByType("program_start").length === 0) {
        const sb = this.workspace.newBlock("program_start");
        sb.initSvg();
        sb.render();
        sb.moveBy(30, 30);
      }
    }

    refreshSelect();
  },

  destroyed() {
    if (this.workspace) {
      this.workspace.dispose();
      window.__blocklyWorkspace = null;
    }
  },
};

// Helper: toggle Run/Stop button visibility
function setRunning(running) {
  const runBtn = document.getElementById("run-btn");
  const stopBtn = document.getElementById("stop-btn");
  if (runBtn) runBtn.classList.toggle("hidden", running);
  if (stopBtn) stopBtn.classList.toggle("hidden", !running);
}

// Helper: append to the log panel
function appendLog(text) {
  const container = document.getElementById("blocks-log-entries");
  if (!container) return;
  const div = document.createElement("div");
  div.className = "py-0.5";
  div.textContent = text;
  container.prepend(div);
  // Keep max 50 entries
  while (container.children.length > 50) {
    container.removeChild(container.lastChild);
  }
}

export const RunBlockly = {
  mounted() {
    this.el.addEventListener("click", () => this.runProgram());
  },

  async runProgram() {
    const workspace = window.__blocklyWorkspace;
    if (!workspace) return;

    const startBlocks = workspace.getBlocksByType("program_start");
    if (startBlocks.length === 0) return;

    // init must be called before blockToCode to set up the generator's state
    javascriptGenerator.init(workspace);
    const code = javascriptGenerator.blockToCode(startBlocks[0]);
    if (!code.trim()) return;

    const robot = createRobotAPI(this);
    window.__currentRobot = robot;

    setRunning(true);
    appendLog("Program started...");
    appendLog("---");

    try {
      const fn = new Function("robot", `return (async () => {\n${code}\n})()`);
      await fn(robot);
      appendLog("---");
      appendLog("Program finished.");
    } catch (e) {
      if (e.message !== "__PROGRAM_STOPPED__") {
        appendLog(`Error: ${e.message}`);
        console.error("Blockly program error:", e);
      } else {
        appendLog("Program stopped.");
      }
    } finally {
      this.pushEvent("stop", {});
      setRunning(false);
      window.__currentRobot = null;
    }
  },
};
