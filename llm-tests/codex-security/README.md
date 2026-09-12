# codex-security "plugin"


```bash
mkdir /tmp/codex-security
cd $_
cat << EOF > package.json
{
  "dependencies": {
    "@openai/codex-security": "^0.1.27"
  }
}
EOF
npm install
cd node_modules/@openai/codex-security/_bundled_plugin
tar -czf claude-codex-security-setup.tar.gz skills references scripts preflight

cd ~/.claude
tar -xvzf /path/to/claude-codex-security-setup.tar.gz
```

This test is what output does the LLM + codex-security components using claude code produce. Using the `skills`, `scripts`, `references`, and `preflight` from the installation of codex-security. Each LLM reviewed the code it generated. (just because)

Results
- LLM: deepseek-v4-flash, OUTPUTS: codex-security-scans-dv4f, dv4f-claude-codex-runtime.txt
- LLM: qwen3.8-27b-uncensored, OUTPUTS: codex-security-scans-qwen3.8-27b-uncensored, qwen3.8-27b-uncensored-runtime.txt
- LLM: qwen3.8-27b-fn, OUTPUTS: codex-security-scans-qwen3.8-27b-fn, qwen3.8-27b-fn-claude-code-runtime.txt