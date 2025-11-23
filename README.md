# LLM Compiler Error Judge

Este projeto implementa um sistema de avaliação automática para mensagens de erro de compiladores. O objetivo é determinar se uma mensagem de erro gerada por um "compilador de estudante" é semanticamente equivalente à mensagem de erro de referência (*gold standard*), mesmo que a fraseologia seja diferente.

O sistema utiliza LLMs locais (via **Ollama**) e técnicas de NLP para atuar como juiz, classificando pares de erros como equivalentes (`True`) ou distintos (`False`).

## 🏆 Melhores Resultados (Destaque)

A arquitetura vencedora utilizou execução paralela com o modelo **Qwen 2.5 3B**.

**Performance de Execução:**
- **Script:** `2_judge_pairs_parallel.py`
- **Tempo Total (754 pares):** ~107s
- **Latência Média:** ~142ms por par
- **Throughput:** Processamento altamente eficiente via *Threading*.

**Métricas de Classificação:**
- **Acurácia:** 85.9%
- **F1-Score (Classe True):** 0.849
- **F1-Score (Classe False):** 0.868

---

## 🧪 Experimentos e Abordagens

O projeto explora quatro estratégias diferentes para resolver o problema de verificação de equivalência de erros:

### 1. Abordagem Paralela (Vencedora)
- **Arquivo:** [`2_judge_pairs_parallel.py`](2_judge_pairs_parallel.py)
- **Lógica:** Utiliza `ThreadPoolExecutor` para enviar múltiplas requisições simultâneas ao servidor do Ollama.
- **Vantagem:** Maximiza o uso da GPU e reduz drasticamente o tempo ocioso do Python esperando I/O. Foi a abordagem mais rápida e estável.

### 2. Abordagem em Lote (Batched)
- **Arquivos:** [`2_judge_pairs_batched.py`](2_judge_pairs_batched.py) e [`2_judge_pairs_batched_v2.py`](2_judge_pairs_batched_v2.py)
- **Lógica:** Agrupa múltiplos pares (ex: 8 ou 16) em um único prompt gigante e pede ao LLM para retornar um JSON com as respostas.
- **Desafio:** Embora reduza o overhead de HTTP, o modelo às vezes falha em formatar o JSON corretamente ou perde a atenção em contextos longos.

### 3. Abordagem Sequencial (Baseline)
- **Arquivo:** [`2_judge_pairs.py`](2_judge_pairs.py)
- **Lógica:** Itera sobre o dataset um por um, enviando uma requisição por vez.
- **Uso:** Serve como linha de base para medir o ganho de velocidade das outras abordagens. É robusta, mas lenta.

### 4. Abordagem via Embeddings (Sem LLM)
- **Arquivo:** [`sentence_transform.py`](sentence_transform.py)
- **Lógica:** Utiliza `SentenceTransformers` (ex: `all-MiniLM-L6-v2`) para gerar vetores numéricos das frases e calcula a Similaridade de Cosseno.
- **Vantagem:** Extremamente rápida (milissegundos).
- **Desvantagem:** Tende a ter menor acurácia em distinções técnicas sutis (ex: confundir "EOF" com "EOL") que o LLM consegue captar via *Few-Shot Prompting*.

---

## 📂 Estrutura do Pipeline

1.  **Construção do Dataset (`0_build_gold.py`)**:
    Extrai casos de teste de arquivos YAML dentro de um zip (`testslogcomp.zip`) para criar o arquivo `gold.jsonl`.

2.  **Geração de Dados Sintéticos (`1_generate_synthetic.py`)**:
    Usa um LLM para criar variações das mensagens de erro:
    - **Pares Positivos:** O LLM parafraseia o erro original (simulando um aluno).
    - **Pares Negativos:** O script mistura erros aleatórios de outros testes.
    - Saída: `synthetic.jsonl`.

3.  **Julgamento (`2_judge_*.py`)**:
    Executa uma das estratégias de julgamento descritas acima. Gera o arquivo `judgments.jsonl`.

4.  **Avaliação (`3_eval_judge.py`)**:
    Compara as previsões do modelo com os rótulos reais, gerando Matriz de Confusão, Acurácia, Precisão, Recall e F1.

## 🛠️ Pré-requisitos e Instalação

1.  **Python 3.8+**
2.  **Ollama** instalado e rodando localmente.
3.  **Modelo Qwen**:
    ```bash
    ollama pull qwen2.5:3b-instruct
    ```
4.  **Dependências Python**:
    ```bash
    pip install requests pyyaml numpy scikit-learn sentence-transformers
    ```

## 🚀 Como Executar (Reproduzindo o Melhor Resultado)

1.  **Gerar os dados:**
    ```bash
    python 0_build_gold.py
    python 1_generate_synthetic.py
    ```

2.  **Rodar o Juiz Paralelo:**
    ```bash
    python 2_judge_pairs_parallel.py
    ```

3.  **Verificar Métricas:**
    ```bash
    python 3_eval_judge.py
    ```

## 🧠 Engenharia de Prompt

Os prompts estão centralizados em [`prompts.py`](prompts.py). Utilizamos **Few-Shot Prompting**, fornecendo ao modelo exemplos de julgamentos corretos (ex: explicando que "token" e "símbolo" são sinônimos, mas "EOF" e "EOL" são diferentes) antes de pedir a classificação atual.