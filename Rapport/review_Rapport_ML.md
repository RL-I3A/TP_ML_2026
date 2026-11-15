# Review of Rapport_ML.pdf

Reviewed all eight PDF pages, including the rendered layout. Page numbers below refer to the PDF. The PDF itself has not been modified.

## Corrections to prioritize before submission

1. **Page 3: inconsistent baseline test values.** The sentence with loss 3.7955, perplexity 44.50 and log-perplexity 5.48 is inconsistent with the displayed baseline training history and the baseline in Table 4. For the evaluated `1Baseline_Model`, use test loss 3.8034, perplexity 44.85, log-perplexity 5.49 bits (5.4872 at four decimals). If another checkpoint is intended, identify it explicitly and use its results consistently.
2. **Pages 6–8: missing similarity evaluation.** Task 5.3 requires predictive performance, embedding quality on the similarity task, and learning speed. The current Results and Table 5 omit similarity, while the conclusion claims a similarity advantage. Restore the method and results: A, Run 2 = 60/97 = 61.86%; B = 89/97 = 91.75%; difference = 29.90 percentage points. The same 97 exact-match triplets are evaluated for both; three containing the absent word `armateur` are excluded.
3. **Page 1: outdated title date.** Replace “September 18, 2023” with the actual report/submission date.
4. **Page 2: maximum versus actual epochs.** Change “A Training contains 50 epochs” to “Training is configured for a maximum of 50 epochs.” Table 1 should say “Maximum epochs.” The illustrated run actually stops after seven epochs.
5. **Page 1: tokens are not necessarily words.** Remove `(word)` after “next token” and `(words)` after the total token count. The BPE count includes word fragments and sentence markers. Write English thousands separators: 5,847,924 and 12,000.
6. **Page 2: quantify repetitions.** State the actual number of baseline runs. Two different baseline CSV histories are currently available locally. Explain that Table 2, Figure 1, the fill-in-the-blank results and the displayed generated sample concern a particular checkpoint; they are not training-run averages.
7. **Page 6: justify the representative A run.** State that Run 2 was selected because its best validation loss is closest to the four-run mean. Do not imply it was selected for its test performance.
8. **Page 7: qualify speed.** State that speed is assessed in completed epochs, not elapsed time: wall-clock times were not recorded, and the cost of obtaining pretrained vectors is excluded.
9. **Page 5: avoid a formal equivalence claim.** Replace “they do not reach a performance equivalent to…” with “the results do not demonstrate equivalent performance” or simply state that every observed run has higher perplexity. The hypothesis's word “close” has no predefined numerical margin.
10. **Pages 7–8: acknowledge one B run.** The results concern one B run and one representative A run in the second comparison. Repeating B remains necessary to estimate its variability; the present result does not establish general or statistically significant superiority.
11. **Whole report: sources are absent.** No references section or in-text citations appear in the PDF. Cite the relevant course/assignment documents and any adopted external methodological references. The normalized-rank definition is explicitly in `donnees_tp.pdf`, page 14. A bibliography command alone does not normally print uncited references.

## Language corrections by page

| PDF page | Current text | Correction |
|---|---|---|
| 1 | token initializations | embedding initialization and trainability |
| 1 | next token (word) | next token |
| 1 | an embedding in form of a numerical vector | an embedding in the form of a numerical vector. |
| 1 | Our initial starting model | Our baseline model |
| 1 | This model is then compared with two research directions | Starting from this baseline, we explore two research directions |
| 1 | improve the model performance | improve model performance |
| 1 | Data and Prepocessing | Data and Preprocessing |
| 1 | The used data for this project consists of french literary texts | The dataset used in this project consists of French literary texts |
| 1 | 5.847.924 tokens (words) | 5,847,924 BPE tokens |
| 1 | 12.000 tokens | 12,000 tokens |
| 1 | The tokens are segmented into subword tokens | The text is segmented into subword tokens |
| 1 | The tokens first get split into sequences of tokens containing… | We construct examples containing three context tokens followed by one target token. |
| 1 | one token at a time which produces | one token at a time, producing |
| 2 | A linear transformation followed by a ReLU activation form the hidden layer with 256 dimensions. | A linear transformation followed by a ReLU activation produces a 256-dimensional hidden representation. |
| 2 | which define a a list of probabilities | which define a probability distribution |
| 2 | The parameters of the initial training are set as in Table 1. | The training hyperparameters are listed in Table 1. |
| 2 | A Training contains 50 epochs. | Training is configured for a maximum of 50 epochs. |
| 2 | In table 2 we can see… | Table 2 presents… |
| 2 | The lowest validation loss of 3.8002, is reached | The lowest validation loss of 3.8002 is reached |
| 2 | early stopping appears | early stopping terminates training |
| 2–3 | which is a phenomen of overfitting: The model is learnig… | Replace the redundant fragment with the complete overfitting passage below. Correct spellings would be “phenomenon” and “learning”. |
| 4 | We generate three samples to see the baseline's performance. | We generate three samples to obtain a qualitative impression of the baseline's behaviour. |
| 4 | three start tokens | three sentence-start tokens |
| 4 | the next token is chosen from the probability distribution | the next token is sampled from the predicted probability distribution |
| 4 | is not consistent grammatically | does not form a consistently grammatical sentence |
| 4 | errors that are not directly seen in the metrics | errors that are not directly captured by the quantitative metrics |
| 4 | learning… / Comparing… in adjacent bullets | Use parallel capitalization and punctuation: “Learning…” / “Comparing…”, both ending with full stops. |
| 5 | initialize the embeddings randomly and make it trainable | initialize the embedding matrix randomly and make it trainable |
| 5 | Adam updates its entries with the weights and biases of the linear layers before. | Adam updates its entries jointly with the weights and biases of the linear layers during training. |
| 5 | We repeat the trainings several times | We perform four training runs |
| 5 | In table 4 we can see… | Table 4 compares… |
| 5 | basline | baseline |
| 5 | compared to the basline model with 44.85 which is a higher perplexity of 11.2% | approximately 11.2% higher than the baseline value of 44.85 |
| 5 | information that is not fully covered through next-token prediction | information that is not fully recovered through next-token prediction |
| 6 | In Addition both models | In addition, both models |
| 6 | On the other hand results support… | The results support our second hypothesis… |
| 6 | learn useful token representations from the prediction network | learn useful token representations through the next-token prediction objective |
| 6 | whether pretrained initialization still perform | whether pretrained initialization remains advantageous |
| 6 | The second one is, that… | Second, we hypothesize that… |
| 6 | initializaion | initialization |
| 6 | random vectors must make useful structure | randomly initialized representations must acquire useful structure |
| 6 | training hyperparameter | training hyperparameters |
| 6 | we reuse a run with randomly initialized runs | we reuse Run 2 with randomly initialized, trainable embeddings |
| 6 | updated by Adam with the parameters of the linear layers | updated by Adam jointly with the parameters of the linear layers |
| 6 | Model A uses random initializations | Model A uses random initialization |
| 6 | Best Epoch | Epoch selected by validation |

## Complete replacement passages

### Page 2: output layer

```latex
A second linear transformation produces 12,000 output
scores. Applying softmax yields a probability distribution
over the next token.
```

This describes the mathematical distribution without claiming that softmax is applied before CrossEntropyLoss in the training code.

### Pages 2–3: overfitting and consistent test values

```latex
The lowest validation loss of $3.8002$ is reached at
epoch two. Subsequently, training loss continues to
decrease while validation loss increases
(Figure~\ref{fig:baseline-loss}), suggesting overfitting.
Early stopping terminates training after epoch seven,
following five epochs without an improvement in
validation loss.

The selected model achieves a test cross-entropy loss
of $3.8034$, corresponding to a perplexity of $44.85$
and a log-perplexity of $5.49$ bits.
```

Adjust the figure label if the actual LaTeX source uses another label.

### Page 5: Adam and repeated runs

```latex
Instead of loading the provided pretrained embeddings,
we initialize the embedding matrix randomly and make
it trainable. Adam updates its entries jointly with
the weights and biases of the linear layers during
training. We perform four training runs with different
random initializations to assess variability between
runs. For each run, we select the model with the
lowest validation loss.
```

### Page 6: two hypotheses

```latex
This research direction investigates whether pretrained
initialization remains advantageous when embeddings
are updated during language-model training.

We formulate two hypotheses. First, pretrained
initialization will lead to lower test perplexity.
Second, it will reach its lowest validation loss at
an earlier epoch than random initialization.
```

### Page 6: similarity method to restore in Implementation

```latex
Embedding quality is assessed using the word triplets
from the previous project. A triplet $(w,w^+,w^-)$ is
successful when
\[
\cos(E(w),E(w^+)) > \cos(E(w),E(w^-)).
\]
Both models are evaluated on the same 97 triplets.
Three triplets are excluded because the word
\emph{armateur} is absent from the vocabulary.
```

### Pages 6–7: similarity results to restore

Add this row to Table 5:

```latex
Similarity accuracy (\%) & 61.86 & 91.75 \\
```

Add an interpretation paragraph:

```latex
Model~B succeeds on 89 of the 97 similarity triplets
($91.75\%$), compared with 60 triplets ($61.86\%$)
for Model~A. The difference is approximately $29.90$
percentage points. Thus, pretrained initialization
also yields better results on this particular
embedding-similarity task after language-model
training. This does not establish superior embedding
quality for all linguistic tasks.
```

### Page 7: speed and repetition limitations

```latex
These observations concern training epochs rather
than elapsed time: no timing measurements were
recorded, and the cost of obtaining the pretrained
vectors is excluded. Moreover, Model~B was evaluated
in a single training run, so its run-to-run
variability remains unknown.
```

## Layout and presentation

- Table 4 on page 5 extends slightly beyond the right text margin. Shorten labels or locally reduce `\tabcolsep` and/or font size. Do not shrink it until it becomes difficult to read.
- The overfitting passage breaks awkwardly across pages 2–3. The shorter replacement above removes most of this problem.
- The long heading on page 6 splits “Initialization”. A shorter heading such as “Effect of Embedding Initialization” is easier to read.
- Pages 4 and 7 have noticeably larger whitespace above the body than several other pages. Check manual spacing, page breaks and float placement; this is cosmetic.
- Table captions could be more informative. For Table 1, use “Baseline architecture and training hyperparameters” and include hidden dimension 256 and optimizer Adam if the table is intended as a complete configuration summary; those values are already present in the surrounding prose.
- The first figure label should be referenced with its actual LaTeX label, not a hardcoded number.

## Confirmed correct / do not change unnecessarily

- Eight pages satisfy the stated 6–10-page template length.
- The normalized-rank formula `(100-r+1)/100` exactly matches `donnees_tp.pdf`, page 14.
- Mean normalized rank 0.9974 corresponds to an average rank of approximately 1.26. It is not top-1 accuracy; the report does not incorrectly equate it with accuracy.
- Table 4's mean and sample SD values are correct.
- The 11.2% perplexity increase versus the fixed baseline and the approximately 21% decrease for B versus A Run 2 are correct.
- Best validation epochs 2 versus 3 are consistent with the CSV histories.
- 12,000 × 32 = 384,000 additional trainable embedding parameters is correct.
- Both figures display properly and their axes/legends are readable.
- Table 1's header is correctly above its rows; text extraction reordered it, but the rendered table is fine.
- Joined words seen in PDF text extraction are not visible spacing errors in the rendered document.
- Figure 2's legend “A” and caption “Run 2” are consistent.
- Do not correct the French grammatical errors inside the generated sample: they are experimental output and are explicitly discussed as such.
