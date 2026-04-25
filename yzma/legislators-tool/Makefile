.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)


.PHONY: build
build: clean ## Build new binary
	@echo "creating new binary"
	@mkdir bin
	@go build -o bin/legislator-tool .

.PHONY: q4
q4: ## Download Q4 quantized version of the LLM 4.06 GB
	@echo "Downloading gemma-4-E4B-it-Q4_K_M.gguf"
	@curl -s -Lo static/gemma-4-E4B-it-Q4_K_M.gguf "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf?download=true"

.PHONY: q8
q8: ## Download Q4 quantized version of the LLM 8.19 GB
	@echo "Downloading gemma-4-E4B-it-Q8_0.gguf"
	@curl -s -Lo static/gemma-4-E4B-it-Q8_0.gguf "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q8_0.gguf?download=true"

.PHONY: clean
clean: ## Delete existing binary
	@echo "cleaning up bin/ folder"
	@rm -rf bin/
