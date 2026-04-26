# Using claude code / agent sdk with llama.cpp
Expose environment variables

```
export ANTHROPIC_AUTH_TOKEN=llama
export ANTHROPIC_BASE_URL=http://llama-cpp.internal:8888
```

## LLMs usage

* gpt-oss-120b (f16, gguf)
* gemma-4


## Example usage

Dumping ground for commands used working with the tool.
### Using claude cli

```
claude --model locall-llm --permission-mode plan

claude --model locall-llm -p "what technologies does this project use?"
claude --model locall-llm  "fix the build error" --allowedTools "Read,Edit,Bash"
claude --model locall-llm -p "Look at the staged changes and create an appropriate commit" \
  --allowedTools "Bash(git diff:*),Bash(git log:*),Bash(git status:*),Bash(git commit:*)" 

claude --model Qwen3-Coder-Next --mcp-config serena.json --teammate-mode in-process

# create a command
mkdir -p .claude/commands
cat > .claude/commands/analyze.md << 'EOF'
# Code Analysis

Analyze the current code for:
- Potential bugs and edge cases
- Performance optimizations
- Code quality improvements
- Security vulnerabilities

Provide specific, actionable recommendations.
EOF

# Worksession
# First request
claude -p "Review this codebase for performance issues"

# Continue the most recent conversation
claude -p "Now focus on the security issues" --continue
claude -p "Generate a summary of all issues found" --continue


```

### using claude agent sdk

Created a K8s job that runs a [code-reviewer](agent-as-job) agent. Using the typescript agent sdk.

```
docker build -t code-reviewer-agent:v1 .

# https://code.claude.com/docs/en/agent-teams
docker run -e ANTHROPIC_AUTH_TOKEN=llama -e ANTHROPIC_BASE_URL=http://192.168.4.24:8888 --rm code-reviewer-agent:v1 npx tsx code-reviewer.ts
```

