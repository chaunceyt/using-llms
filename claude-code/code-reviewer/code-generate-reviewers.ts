// Learning to use the Claude Agent SDK as a requirement for the "Claude Certified Architect - Foundations (CCA-F)"

// Notes:
// An agent is the query loop: prompt -> pick tools -> execute -> observe -> repeat
// Provide the prompt. The SDK handles the loop, allows one to build multi-agents solutions
// Provides access to "open standards" that enhance agent functionality, MCP servers, skills,
// permissions.

import { 
  query,
  type HookCallback,
  type PreToolUseHookInput,
  type PostToolUseHookInput,
  SessionStartHookInput,
  UserPromptSubmitHookInput,

} from "@anthropic-ai/claude-agent-sdk";

const LEARN_CODEBASE_PROMT =   `You have two subagents available: "gemma-4-researcher" and "gemma-4-analyst".

First, use "gemma-4-researcher" to thoroughly explore this project. The researcher should:
- Read every TypeScript file in the directory
- Read every agent definition file in .claude/agents/
- Read the package.json at the project root
- Return full file contents with paths

Then pass ALL of the researcher's findings to "gemma-4-analyst" for a comprehensive architectural analysis.
The analyst should produce at least 600 words covering design patterns, module relationships, strengths, and recommendations.

Use both agents. Be thorough.`

const CODE_REVIEW_PROMPT = `Create an agent team to review the code in this project. Spawn three reviewers:
- One focused on security implications
- One checking performance impact
- One validating test coverage
Have them each review and report findings.`

const USE_SUBAGENT_PROMPT = `You have two subagents available: "gemma-4-researcher" and "gemma-4-summarizer".
Use the "gemma-4-researcher" to find all TypeScript files in this project and read their contents.
Then use the "gemma-4-summarizer" to create a brief summary of the project.
Use both agents. Be thorough.`

const SDK_INVESTIGATION_PROMPT =  `Use "gemma-4-researcher" again to specifically investigate the SDK type definitions.
The researcher should read the following files completely:
- node_modules/@anthropic-ai/claude-agent-sdk/sdk.d.ts  (the full type declaration file)

Then use "gemma-4-analyst" to analyze the SDK's type system: what types are exported, how the
session API differs from the query API, what hook types are available, and how ModelUsage
tracks context window data. The analyst should provide a thorough comparison with code examples.`



// taken from: https://github.com/grmkris/claude-agent-sdkexploration/blob/main/examples/06-hooks.ts
// Hook: Log every tool call
const toolStartTimes = new Map<string, number>();
const auditLog: HookCallback = async (input, toolUseID, { signal }) => {
  if (input.hook_event_name === "PreToolUse") {
    const pre = input as PreToolUseHookInput;
    toolStartTimes.set(pre.tool_use_id, Date.now());
    console.log(`  [AUDIT] Tool start: ${pre.tool_name} | input: ${JSON.stringify(pre.tool_input).slice(0, 80)}`);
  } else if (input.hook_event_name === "PostToolUse") {
    const post = input as PostToolUseHookInput;
    const startTime = toolStartTimes.get(post.tool_use_id);
    const elapsed = startTime ? `${Date.now() - startTime}ms` : "unknown";
    toolStartTimes.delete(post.tool_use_id);
    console.log(`  [AUDIT] Tool done: ${post.tool_name} | elapsed: ${elapsed}`);
  }
  return {};
};

const logHooks: HookCallback = async (input, toolUsedID, { signal }) => {
  if (input.hook_event_name === "SessionStart") {
    const ss = input as SessionStartHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${ss.hook_event_name} | input: ${JSON.stringify(ss.source).slice(0, 80)}`);  
  }

  if (input.hook_event_name === "UserPromptSubmit") {
    const up = input as UserPromptSubmitHookInput;
    console.log(`  [LOG_HOOKS] Hook: ${up.hook_event_name} | input: ${JSON.stringify(up.prompt).slice(0, 80)}`);  
  }
  return {};
}

async function codeReview(prompt: string) {
 let turnCount = 0;

for await (const message of query({
  prompt: prompt,
  options: {
    allowedTools: ["Read", "Grep", "Glob", "Agent", "Task"],
    model: "gemma-4",
    settingSources: ['project'],
    hooks: {
      PreToolUse: [
        { hooks: [auditLog, logHooks] },
       ],
      PostToolUse: [{ hooks: [auditLog] }],
    },
    // mcpServers: {
    //   "aws-knowledge-mcp-server": {
    //         "command": "uvx",
    //         "args": ["fastmcp", "run", "https://knowledge-mcp.global.api.aws"]
    //   }
    // },
  }
})) {

    if (message.type === "system") {

      if (message.subtype === "init") {
        const servers = message.mcp_servers ?? []
        
        console.log("\n=== Code Reviewer Agent ===\n");
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
          if (block.name === "Task") {
            console.log(`🤖 Delegating to: ${(block.input as any).subagent_type}`);
            console.log(`\n[tool_use] ${block.name}(${JSON.stringify(block.input).slice(0, 300)})`);

          } else {
            console.log(`\n[tool_use] ${block.name}(${JSON.stringify(block.input).slice(0, 300)})`);
          }

        }
      }
    }  
      // Show completion status
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

codeReview(LEARN_CODEBASE_PROMT)
// codeReview(CODE_REVIEW_PROMPT)
// codeReview(USE_SUBAGENT_PROMPT)
// codeReview(SDK_INVESTIGATION_PROMPT)
