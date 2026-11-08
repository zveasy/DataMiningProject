# Dropout on MNIST · 18-minute presentation

| Slide | Time | On screen | Code and evidence |
|---|---|---|---|
| 1. Research question | 0:00–0:45 | Which drop probability generalizes best on handwritten digits? | README and experiment_summary.md |
| 2. Overfitting | 0:45–2:00 | Trace divergence between train and validation loss | src/train.py · learning_loss.png |
| 3. The 2014 study | 2:00–3:30 | Paper Table 2 and the scope of this comparison | README paper discussion · src/model.py |
| 4. Sample a subnetwork | 3:30–5:15 | Bernoulli mask, keep probability and inverted scaling | src/model.py · tests/test_model.py |
| 5. MNIST and the split | 5:15–6:30 | A digit from each class, followed by the stratified counts | src/dataset.py · examples.png · distribution.png |
| 6. Model architecture | 6:30–7:30 | 784 → 512 → 256 → 10, with two ReLU/dropout stages | src/model.py · DigitNet |
| 7. Comparison protocol | 7:30–9:00 | Three paired seeds, fixed epochs, minimum validation loss | configs/experiment.json · src/train.py |
| 8. Code walkthrough | 9:00–10:15 | Trace a minibatch through fit, then inspect score | src/train.py · fit and score |
| 9. Results | 10:15–12:45 | Start with test accuracy, then show gaps and the size study | experiment_summary.md · comparison.png · gap.png · size.png |
| 10. Errors | 12:45–14:15 | Read a directional confusion count and inspect examples | src/evaluate.py · confusion.png · misclassified.png |
| 11. Live inference | 14:15–16:45 | Run Draw a digit twice and compare all three distributions | notebooks/live_demo.ipynb |
| 12. Conclusion and references | 16:45–18:00 | Answer the question within the study design | experiment_summary.md · sensitivity.png · README references |

Figures are under `results/figures/`. Keep the remaining two minutes for questions or a slow screen transition.
