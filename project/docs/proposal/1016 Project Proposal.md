# **Beyond the Dyad: Modeling Linguistic Style Transfer Across Conversational Partners**

**Eric Wu, Hao Su, Luis Zhang**

### **Research Question and Hypotheses**

When people adapt their linguistic style during conversations, do they learn separately from each partner, or does experience with one partner influence communication with others? Building on Kronmüller and Guerra (2026) and Hawkins et al. (2023), we investigate whether linguistic adaptation is partner-specific, context-general, or governed by a hierarchical combination of both.

Our central hypothesis is that listeners maintain both shared and partner-specific representations of linguistic style. A hierarchical Bayesian model allowing partial pooling should therefore better predict subsequent language production than models assuming either completely independent or fully shared learning.

### **Task, Stimuli, and Data**

We will analyze the publicly available Diplomacy corpus (Peskov et al., 2020), containing **17,289 messages across 12 seven-player games**. Each message identifies its sender, recipient, text, and position in the game's message sequence.

Each outgoing message from player P to partner B constitutes a prediction trial. Using only messages observed before that trial, our models predict the linguistic style of P's next message to B.

Rather than generating text, we represent each message as a multidimensional style vector containing measurable features, such as pronouns, hedging expressions, modal constructions, syntactic patterns, contractions, and punctuation. Game-specific content will be masked, and features will be normalized to reduce topic and message-length confounds.

The primary behavioral measure is the observed style vector of each subsequent message.

### **Computational Models**

We will implement four competing models:

1. **Baseline Model:** Predicts P's style using their individual baseline and observable conversational context, without learning from incoming messages.  
2. **Partner-Specific Bayesian Model:** Maintains independent Gaussian style expectations for each partner, updated only through that partner's messages.  
3. **Fully Pooled Bayesian Model:** Maintains one shared Gaussian style expectation, updated using incoming messages from all partners.  
4. **Hierarchical Bayesian Model:** Assumes partner-specific style expectations are drawn from a shared Gaussian distribution, allowing partial pooling of linguistic experience across partners.

All four models use a common probabilistic observation model for predicting continuous style features. The hierarchical model includes a pooling parameter controlling cross-partner generalization and an adaptation coefficient determining how strongly P's output shifts toward learned expectations.

### **Experimental Manipulation and Model Comparison**

We will evaluate the models against actual message-level behavioral data from the Diplomacy corpus, processing messages chronologically to prevent future-information leakage.

Our primary comparison will measure **held-out predictive log likelihood**, aggregated at the game level. We will test whether incorporating messages from third-party speakers improves predictions of P's subsequent messages to B beyond P's own baseline and B's prior messages.

To isolate the models' theoretical predictions, we will also conduct controlled simulations. Holding B's message history and the conversational context constant, we will vary the style and source of messages received from A. A partner-specific model predicts no transfer to B, a fully pooled model predicts generalized adaptation, and a hierarchical model predicts partial transfer moderated by prior experience with B.

We will additionally use messages not observed by P as negative controls to distinguish exposure-based predictions from shared conversational context.

### **Expected Contribution**

This project tests whether human linguistic adaptation relies on independent partner representations, shared contextual learning, or hierarchical generalization. By comparing computational predictions with natural human conversations, we aim to identify which cognitive learning mechanism best accounts for partner-specific alignment and cross-partner style transfer.

**References:** Kronmüller & Guerra (2026); Hawkins et al. (2023); Peskov et al. (2020).

