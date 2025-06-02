# 📊 Samsung Mobile Sales - Análise e Modelagem

Este projeto realiza uma análise exploratória e modelagem preditiva com dados de vendas de celulares Samsung. O dataset foi obtido do [KaggleHub](https://www.kaggle.com/datasets/datatechexplorer/samsung-mobile-sales-dataset).

Seu objetivo principal foi construir um modelo preditivo para estimar o número de unidades vendidas de celulares Samsung, com base em variáveis de mercado, indicadores de adoção de tecnologia 5G e atributos do produto.

---


## 📁 Estrutura Esperada

- `Expanded_Dataset.csv` (baixado automaticamente via `kagglehub`)
- Código estruturado em células com seções bem definidas

---

## 📌 Requisitos

- Python 3.x
- Bibliotecas: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `xgboost`, `kagglehub`

Instalar com:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib xgboost kagglehub
```

---

## 🚀 Etapas do Projeto

### 1. 📦 Importação de Bibliotecas
Utilizamos bibliotecas como:
- `pandas`, `numpy` para manipulação de dados
- `seaborn`, `matplotlib` para visualizações
- `scikit-learn`, `xgboost` para modelagem
- `kagglehub` para baixar o dataset

---

### 2. 📥 Carregamento do Dataset
O dataset `Expanded_Dataset.csv` é carregado automaticamente via `kagglehub`.

---

### 3. 🔍 Análise Exploratória Inicial
- Visualização dos primeiros e últimos registros
- Estrutura da tabela (`df.info()`)
- Estatísticas descritivas (`df.describe()`)
- Verificação de valores nulos

---

### 4. 🔢 Cardinalidade de Atributos Categóricos
- Verifica o número de valores únicos por coluna categórica
- Visualização com gráfico de barras (`seaborn.barplot`)

---

### 5. 📊 Frequência de Categorias
- Visualizações `countplot` das colunas com baixa cardinalidade
- Separação automática de colunas com alta/baixa cardinalidade

---

### 6. 📈 Distribuição de Vendas
Histograma de distribuição da coluna `Units Sold`.

---

### 7. 🧪 Engenharia de Atributos
- **Count Encoding** na coluna `Product Model`
- **One-Hot Encoding** nas colunas categóricas com baixa cardinalidade
- Definição de `X` (features) e `y` (target)

---

### 8. 🔀 Divisão Treino/Teste
70% dos dados para treino e 30% para teste.

---

### 9. ⚙️ Construção de Pipeline
- Aplicação de `StandardScaler` para atributos numéricos
- Uso de `Pipeline` com `ColumnTransformer`
- Modelo base: `RandomForestRegressor`

---

### 10. 🧠 Treinamento do Modelo
Treinamento completo usando a pipeline definida.

---

### 11. 📐 Avaliação do Modelo
- Métricas: `R²`, `RMSE`, `MAE`
- Gráfico de previsão vs. valores reais

---

### 12. 🔍 Busca de Hiperparâmetros (GridSearchCV)
- Teste de combinações de `n_estimators`, `max_depth`, `min_samples_split`
- Melhor modelo é avaliado novamente com as mesmas métricas

---

### 13. 📊 Interpretação: Importância dos Atributos
- Extração da importância das variáveis a partir da `RandomForest`
- Visualização dos 10 principais atributos

---

## 📌 Conclusão

### 1. Desempenho do Modelo Baseline

- **R² no teste:** 0.7356  
- **RMSE no teste:** 8 297,48 unidades  
- **MAE no teste:** 5 195,27 unidades  

Esses valores indicam que o modelo Random Forest baseline consegue explicar cerca de 73,56 % da variabilidade nas vendas de celulares Samsung, apresentando um **erro médio absoluto** de aproximadamente 5 195 unidades. Em um cenário de vendas em larga escala, onde volumes podem chegar a milhões de unidades, esse nível de precisão é razoavelmente bom, pois reduz a incerteza das previsões em um patamar gerenciável.  

Adicionalmente, quando representamos graficamente (imagem abaixo) o ajuste entre valores reais e previstos, observa-se que a maior parte dos pontos está concentrada ao redor da linha de identidade (y = x), sugerindo que não há grandes desvios sistemáticos (viés) em uma faixa ampla de valores. Eventuais outliers — vendas extremamente altas ou baixas — podem ainda gerar erro localizado, mas não comprometem o comportamento geral do modelo.

![Captura de tela de 2025-06-02 02-38-27](https://github.com/user-attachments/assets/3d3540e9-7840-4715-b54b-e824332d14e5)


### 2. Otimização de Hiperparâmetros (GridSearchCV)

Ao explorar diferentes combinações de parâmetros para a Random Forest, os melhores resultados de validação cruzada foram obtidos com:
```json
{
  "model__n_estimators": 200,
  "model__max_depth": null,
  "model__min_samples_split": 2
}
```
Entretanto, quando avaliamos esse modelo otimizado no conjunto de teste final, os resultados foram:

R² após GridSearch: 0.7291

RMSE no teste: 8 398,72 unidades

MAE no teste: 5 250,21 unidades

Note que o R² caiu ligeiramente de 0.7356 para 0.7291, e tanto o RMSE quanto o MAE aumentaram de forma marginal em comparação ao modelo baseline. Isso sugere que o pipeline original (com 100 árvores e sem limite de profundidade) já estava próximo ao ponto ideal para este conjunto de dados específico. Em outras palavras, aumentar o número de árvores ou ajustar a profundidade mínima de divisão não gerou ganho significativo de performance — possivelmente devido ao fato de que já havia saturação de informações ou que o overfitting começou a aparecer a partir de certo ponto.

Em termos práticos, esse comportamento indica que, para problemas de previsão de vendas baseados em atributos semelhantes, vale a pena testar uma configuração inicial simples de Random Forest e só partir para ajustes mais finos se houver grande quantidade de dados ou sinais de subajuste (underfitting) ou sobreajuste (overfitting) evidentes.

### 3. Importância das Variáveis (Feature Importance)
A análise de importância das variáveis, extraída diretamente do melhor modelo Random Forest, mostra o peso relativo de cada atributo na predição. As dez features mais importantes foram:

1. Revenue ( $ ) – 0.200706

2. Avg 5G Speed (Mbps) – 0.152817

3. Preference for 5G (%) – 0.141004

4. Regional 5G Coverage (%) – 0.140912

5. 5G Subscribers (millions) – 0.127568

6. Market Share (%) – 0.119789

7. Product_Model_Count – 0.064296

8. Year – 0.052908

![Captura de tela de 2025-06-02 02-42-17](https://github.com/user-attachments/assets/a132f325-7b36-4aca-98f9-a0dc87744d41)

Observa-se que as métricas relacionadas à tecnologia 5G (velocidade média, preferência do usuário, cobertura regional e número de assinantes) ocupam as quatro posições de maior relevância somadas ao quinto lugar, o que representa mais de 56 % da importância total atribuída pelo modelo. Esse resultado reforça que, em 2025, a adoção do 5G é um dos principais direcionadores das vendas de smartphones: onde a infraestrutura está mais madura e a preferência do consumidor é maior, há, em média, uma demanda elevada por dispositivos compatíveis.

Em seguida, vemos o Revenue ($) e o Market Share (%), que somam cerca de 32 % de importância conjuntural. Isso significa que, além da tecnologia, o desempenho econômico da marca (receita gerada e fatia de mercado) também exerce forte influência sobre o volume vendido.
Por fim, o Product_Model_Count (que reflete a frequência histórica de um modelo específico) e o Year (que capta tendências temporais) aparecem com importâncias menores, mas ainda assim relevantes para refinar as previsões — indicando que modelos mais populares ou o próprio contexto temporal (lançamentos recentes, sazonalidades) têm impacto secundário, porém distinto.

### 4. Lições Aprendidas e Recomendações Futuras
#### Simplicidade do Pipeline
O fato de a configuração padrão de Random Forest apresentar desempenho muito próximo ao do modelo otimizado demonstra que, muitas vezes, iniciar com hiperparâmetros padrões (n_estimators = 100, sem limite de profundidade) já é suficiente para capturar grande parte da variabilidade em problemas de regressão com dados estruturados.

#### Importância do 5G no Mercado Mobile
A preponderância de variáveis associadas ao 5G indica que, para empresas que atuam nesse setor, monitorar continuamente indicadores como cobertura regional, velocidade média e número de assinantes é crucial para ajustar estoques, lançar campanhas direcionadas e programar lançamentos de forma estratégica.

#### Variáveis Macro e Dados Complementares
Apesar de métricas de mercado (Revenue e Market Share) terem se mostrado relevantes, vale a pena incorporar, em análises futuras, indicadores macroeconômicos (PIB, taxa de câmbio, inflação) ou dados de concorrentes (lançamentos, preços médios). Isso pode ajudar a entender melhor cenários de instabilidade ou mudanças abruptas de consumo.

#### Testar Outros Algoritmos e Validações
Embora a Random Forest tenha se saído bem, explorar modelos baseados em gradiente (XGBoost, LightGBM) ou, em outra perspectiva, redes neurais leves (MLP) pode revelar ganhos adicionais, especialmente se o conjunto de dados crescer em volume ou se forem incluídas novas variáveis (como imagens dos aparelhos, especificações detalhadas, reviews de usuários).

#### Monitoramento Contínuo e Atualização de Dados
Em mercados dinâmicos como o de smartphones, a importância de cada variável pode mudar rapidamente. Portanto, implementar um processo de retraining periódico (mensal ou trimestral) garantirá que o modelo reflita as tendências atuais (novos padrões de consumo, mudanças de preço, inovações tecnológicas).

### Conclusão Final:
O modelo desenvolvido demonstrou-se robusto e estável, capaz de explicar mais de 73 % da variabilidade nas vendas de celulares Samsung com base em métricas de adoção de 5G e indicadores de mercado. A otimização de hiperparâmetros não trouxe ganhos substanciais, reforçando a eficácia de uma abordagem inicial simples. A ênfase em variáveis relacionadas ao 5G evidencia a transformação tecnológica do setor e serve como guia para decisões estratégicas de produto e marketing. Por fim, recomenda-se ampliar o escopo de variáveis (macro e concorrência), testar outros algoritmos e estabelecer ciclos regulares de atualização do modelo para manter a acurácia em ambientes voláteis.

---


## 📎 Fonte

Notebook original: [Google Colab](https://colab.research.google.com/drive/1Ork4XYaSPfW9WrITlC0HjpxNO5jOCF4G)

Dataset: [Samsung Mobile Sales Dataset](https://www.kaggle.com/datasets/datatechexplorer/samsung-mobile-sales-dataset)

Projeto: [Projeto Base](https://www.kaggle.com/code/alicanpayasli/samsung-mobiles-eda-feature-eng-models)

---

## 📬 Contato

Contribuições, dúvidas ou sugestões? Fique à vontade para abrir uma issue ou entrar em contato!

