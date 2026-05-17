// learning hook system.
// https://code.claude.com/docs/en/agent-sdk/hooks
import { 
  query, 
  tool, 
  createSdkMcpServer, 
  type HookCallback,
  type PreToolUseHookInput,
  type PostToolUseHookInput,
  type PostToolUseFailureHookInput,
  type PermissionRequestHookInput,
  type PermissionDeniedHookInput,
  type SubagentStartHookInput,
  type SubagentStopHookInput,
  type SessionEndHookInput,
  type TaskCreatedHookInput,
  type TaskCompletedHookInput,
  type FileChangedHookInput,
  type CwdChangedHookInput,
  type AgentDefinition,
  SessionStartHookInput,
  UserPromptSubmitHookInput,

} from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

type ToolResult = { content: [{ type: "text"; text: string }] };

function toolResult(data: unknown): ToolResult {
  return { content: [{ type: "text" as const, text: JSON.stringify(data, null, 2) }] };
}

function toolError(error: unknown): ToolResult {
  const message = error instanceof Error ? error.message : String(error);
  return { content: [{ type: "text" as const, text: JSON.stringify({ error: message }) }] };
}

// Functions for each tool

export async function listEventsHandler(args: { namespace: string }): Promise<ToolResult> {
  try {
    return toolResult({ namespace: args.namespace, status: "exists" });
  } catch (error) {
    return toolError(error);
  }
}

export async function getResourcesHandler(args: { namespace: string, podname: string}): Promise<ToolResult> {
  try {
    return toolResult({ namespace: args.namespace, podname: args.podname, status: "exists" });
  } catch (error) {
    return toolError(error);
  }
}

export async function listResourcesHandler(args: { namespace: string, podname: string}): Promise<ToolResult> {
  try {
    return toolResult({ namespace: args.namespace, podname: args.podname, status: "exists" });
  } catch (error) {
    return toolError(error);
  }
}

export async function analyzeHandler(args: { namespace: string, podname: string}): Promise<ToolResult> {
  try {
    return toolResult({ namespace: args.namespace, podname: args.podname, status: "exists", data: "analyze cluster" });
  } catch (error) {
    return toolError(error);
  }
}

// Tools

const listEvents = tool(
  "list_events",
  "Get recent events in the cluster.",
  { namespace: z.string().describe("Namespace") },
  listEventsHandler,
);

const getResources = tool(
  "get_resources",
  "Get pod details and check status, conditions, and events.",
  { 
     namespace: z.string().describe("Namespace"),
     podname: z.string().describe("Pod Name"),
  },
  getResourcesHandler,
);

const listResources = tool(
  "list_resources",
  "Get resourceTypes (pods, replicasets) details and check status, conditions, and events.",
  { 
     namespace: z.string().describe("Namespace"),
     podname: z.string().describe("Pod Name"),
  },
  listResourcesHandler,
);

const analyze = tool(
  "analyze",
  "Perform a comprehensive analysis",
  { 
     namespace: z.string().describe("Namespace"),
     podname: z.string().describe("Pod Name"),
  },
  analyzeHandler,
);


export const serverTools = [getResources, listEvents, listResources, analyze];


// Create custom MCP server
const customServer = createSdkMcpServer({
  name: "cluster-ops",
  version: "1.0.0",
  tools: serverTools,
});

// taken from: https://github.com/grmkris/claude-agent-sdkexploration/blob/main/examples/06-hooks.ts
// Hook: Log every tool call
const toolStartTimes = new Map<string, number>();
const auditLog: HookCallback = async (input, toolUseID, { signal }) => {
  if (input.hook_event_name === "PreToolUse") {
    const pre = input as PreToolUseHookInput;
    toolStartTimes.set(pre.tool_use_id, Date.now());
    console.log(`  [AUDIT] Tool start: ${pre.tool_name} | input: ${JSON.stringify(pre.tool_input, null, 2)}`);
  } else if (input.hook_event_name === "PostToolUse") {
    const post = input as PostToolUseHookInput;
    const startTime = toolStartTimes.get(post.tool_use_id);
    const elapsed = startTime ? `${Date.now() - startTime}ms` : "unknown";
    toolStartTimes.delete(post.tool_use_id);
    console.log(`  [AUDIT] Tool done: ${post.tool_name} | elapsed: ${elapsed}`);
  }
  return {};
};

// learning about the hooks 

const  sessionStartHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "SessionStart") {
    // initialize logging and telemetry
    const ss = input as SessionStartHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${ss.hook_event_name} | input: ${JSON.stringify(ss.source)}`);  
  }
  return {};
}

const userPromptSubmitHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "UserPromptSubmit") {
    // inject additional context into prompts
    // scan input
    const up = input as UserPromptSubmitHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${up.hook_event_name} | input: ${JSON.stringify(up.prompt)}`);  
  }
  return {};
}

const permissionRequestHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "PermissionRequest") {
    // custom permission handling
    const perm = input as PermissionRequestHookInput;

    console.log(`  [LOG_HOOKS] Hook: ${perm.hook_event_name} | input: ${JSON.stringify(perm.permission_suggestions)}`);  
  }
  return {};
}

const permissionDeniedHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "PermissionDenied") {
    // handle and log event
    const res = input as PermissionDeniedHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res.tool_name)}`);  
  }
  return {};
}

const subagentStartHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "SubagentStart") {
    // track parallel task spawning
    const res = input as SubagentStartHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res, null, 2)}`);  
  }
  return {};
}

const subagentStopHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "SubagentStop") {
    // aggregate results from parallel tasks
    const res = input as SubagentStopHookInput;
    console.log(`[SUBAGENT] Completed: ${input.agent_id}`);
    console.log(`  Transcript: ${input.agent_transcript_path}`);
    console.log(`  Tool use ID: ${toolUsedID}`);
    console.log(`  Stop hook active: ${input.stop_hook_active}`);
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res, null, 2)}`);  
  }
  return {};
}

const preToolUseHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "PreToolUse") {
    // block dangerous shell commands
    // e.g. usage run gitleaks to scan commands before execution
    const res = input as PreToolUseHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res, null, 2)}`);  
  }
  return {};
}

const postToolUseFailureHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "PostToolUseFailure") {
    // handle or log tool errors
    const res = input as PostToolUseFailureHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res, null, 2)}`);  
  }
  return {};
}

const postToolUseHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "PostToolUse") {
    // log all file changes to audit trail
    // after a tool call succeeds
    // before sending to llm transform
    const res = input as PostToolUseHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res, null, 2)}`);  
  }
  return {};
}

const sessionEndHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "SessionEnd") {
    // clean up temporary resources
    const res = input as SessionEndHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res, null, 2)}`);  
  }
  return {};
}

const taskCreatedHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "TaskCreated") {
    const res = input as TaskCreatedHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res, null, 2)}`);  
  }
  return {};
}

const taskCompletedHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "TaskCompleted") {
    // aggregate results from parallel tasks
    const res = input as TaskCompletedHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res, null, 2)}`);  
  }
  return {};
}

const fileChangedHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "FileChanged") {
    const res = input as FileChangedHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res, null, 2)}`);  
  }
  return {};
}

const cwdChangedHook: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "CwdChanged") {
    const res = input as CwdChangedHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${res.hook_event_name} | input: ${JSON.stringify(res, null, 2)}`);  
  }
  return {};
}

// agent

function createSecurityAgent(securityLevel: "basic" | "strict"): AgentDefinition {
  const isStrict = securityLevel === "strict";
  return {
    description: "Security code reviewer",
    // Customize the prompt based on strictness level
    prompt: `You are a ${isStrict ? "strict" : "balanced"} security reviewer...`,
    tools: ["Read", "Grep", "Glob"],
    // Key insight: use a more capable model for high-stakes reviews
    model: isStrict ? "opus" : "sonnet"
  };
}


async function runAgent() {
  for await (const message of query({
    prompt: `We are writing a cluster-ops mcp server that will function as the SRE for the environment. 
     We have a number of tools. Execute each of the tools and return the entire response from each.
     DO NOT use BASH command. We just want the entire response from the cluster-ops MCP server.

     Create two subagents to handle the execution of the tools. One subagent should be responsible for the resources related and 
     the other subagent should handle the analyze and events.

     The subagents shouldn't use BASH command either.

     We just want the entire respeone from the cluster-ops MCP server.

     Additional information that can be used as parameters for the tools.

     <namespaces>
     team-a
     team-b
     </namespaces>

     <podnames>
     team-a-pod
     team-b-pod
     </podnames>
     `,
    options: {
      model: "Qwen3.6-27B-Q8_0",
      settingSources: ['project'],
      
      mcpServers: {
        "cluster-ops": customServer
      },
      agents: {
      // Call the factory with your desired configuration
      "security-reviewer": createSecurityAgent("strict")
      },
      
      // MCP tools follow the pattern: mcp__<server-name>__<tool-name>
      allowedTools: [
        "Task",
        "mcp__cluster-ops__get_resources", 
        "mcp__cluster-ops__list_events",
        "mcp__cluster-ops__list_resources",
        "mcp__cluster-ops__analyze"
      ],
      
      // Maximum number of back-and-forth turns before stopping
      maxTurns: 50,
      hooks: {
        SessionStart: [{ hooks: [sessionStartHook] }],  
        UserPromptSubmit: [{ hooks: [userPromptSubmitHook] }],
        PermissionRequest: [{hooks: [permissionRequestHook]}],
        PermissionDenied: [{hooks: [permissionDeniedHook]}],
        PreToolUse: [{ hooks: [preToolUseHook] }],
        PostToolUse: [{ hooks: [postToolUseHook] }],
        PostToolUseFailure: [{ hooks: [postToolUseFailureHook] }],
        SubagentStart: [{ hooks: [subagentStartHook] }],
        SubagentStop: [{ hooks: [subagentStopHook] }],
        SessionEnd: [{ hooks: [sessionEndHook] }],
        TaskCreated: [{ hooks: [taskCreatedHook] }],
        TaskCompleted: [{ hooks: [taskCompletedHook] }],
        FileChanged: [{ hooks: [fileChangedHook] }],
        CwdChanged:  [{ hooks: [cwdChangedHook] }],
       
    },
    }
  })) {

    if (message.type === "system") {

      if (message.subtype === "init") {
        const servers = message.mcp_servers ?? []
        
        console.log(`[init] session-id: ${message.session_id}\n`);
        console.log(`[init] model: ${message.model}`);
        console.log(`[init] agents: ${message.agents?.join(", ") ?? "none"}`);
        console.log(`[init] tools: ${message.tools.join(", ")}\n`);
        console.log(
          `[init] MCP servers: ${servers.map((s) => `${s.name} (${s.status})`).join(", ") || "none"}`
        )
          // Skills integration: list all available skills at startup
        if (message.skills && message.skills.length > 0) {
          console.log(`[init] skills:         ${message.skills.join(", ")}`);
        } else {
          console.log(`[init] skills:         (none installed)`);
        }
        console.log();        
        
      } else if (message.subtype === "task_started") {
          console.log(
          `\n[task_started] ${message.description} (id: ${message.task_id})`
          );
      } else if (message.subtype === "task_notification") {

          const statusIcon = message.status === "completed" ? "✓" : message.status === "failed" ? "✗" : "■";
          console.log(`\n  ${statusIcon} Task ${message.status}: ${message.summary.slice(0, 180)}${message.summary.length > 180 ? "…" : ""}`);
          if (message.usage) {
            console.log(`    tokens: ${message.usage.total_tokens.toLocaleString()}  tools: ${message.usage.tool_uses}  time: ${(message.usage.duration_ms / 1000).toFixed(1)}s`);
          }        
      } else if (message.subtype === "status" && message.status === "compacting") {
          console.log("\n  ⏳ Context compacting...");
      }

    }

    // Show Claude's analysis as it happens
    if (message.type === "assistant") {

      for (const block of message.message.content) {
        if ("text" in block) {
          console.log(block.text);
        } else if ("name" in block) {
          console.log(`\n Block name: ${block.name}`)
          console.log(`\n[tool_use] ${block.name}(${JSON.stringify(block.input, null, 2).slice(0, 300)})`);
        }
      }
    } 
   
    
    // Handle the final result when the agent loop completes
    if (message.type === "result") {
      if (message.subtype === "success") {
          console.log(`✅ Status: ${message.subtype}`);
          console.log(`�💰 Cost: $${message.total_cost_usd.toFixed(4)}  | Turns: ${message.num_turns}`);
          console.log(
            `📈 Tokens: ${message.usage.input_tokens} in / ${message.usage.output_tokens} out`
          );
      } else {
        console.log(`\n❌ Status: ${message.subtype}`);
      }
    }
  }
}

runAgent();
