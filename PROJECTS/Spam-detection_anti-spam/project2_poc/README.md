# XAI methods for interpreting a spam detection NN model
## Important neurons selection method
- Train first (original) model on normal spam dataset to detect spam/ham.
- For each record in test dataset save each neuron activation value in the end acquiring new dataset with activation values in model and original test dataset labels as labels.
- Train new linear model on a newly created dataset.
- We rank neurons based on absolute value of corresponding weight in linear model (the value of first weight corresponds to first neuron in first original model)
- Evaluate original model performance on original test set with only some part of neurons(by putting 0 as an output of omitted neurons)
- Test if picking neurons from each layer separately affect model performance based on standard model evaluation (picking top 10% of neurons).
- Repeat these steps to achieve statistical analysis

### Word search
- Analyse test dataset for searched keyword (for example word “free”)
- Extract only records with the given word
- For each sentence with a keyword visualise effect of each word 
- Highlight the keyword in each sentence

### Neuron search for given keyword
- Analyse test dataset for searched keyword (for example word “free”)
- Extract only records with the given word
- For a subset of neurons (based on neuron ranking, or for each but it depends on computational power) compute  how much the keyword affects its own (neuron) activation in comparison with other words in sentences
- Pick the most affecting neuron and visualize its effect on sentences with a keyword to try to understand its purpose 

## Frozen weights method
This method consists in describing our model as a vector in an Euclidean space, and projecting it as a linear combination of other vectors, which we are able to associate with basic linguistic properties (in our case, all the properties considered refer to the presence of a specific word in the mail).

### Methodology
- Train first (original) model on normal spam dataset to detect spam/ham
- Extract the inputs/outputs weights of each neuron
- Flatten all the weights into a vector of R^d
- Select a set of words from the vocabulary of the dataset
- Train for each word a NN model (of the same architecture and configuration as the original one) for detecting the occurence of the corresponding word in an e-mail
- Convert each model into a vector of R^d (the same process as for the original model)
- Select the most relevant of these vectors for projection
- Compute the scalar products between the original and each vector of the basis
- The coefficient relative to each word indicates the correlation between the detection of spam mails and the detection of the corresponding specific word

### Selection
#### First selection
A certain number of mails are randomly selected, and for each the important neurons method is computed. The $\rho$ words that have the highest coefficient in absolute value are selected. Then the duplicate words are removed.
#### Second selection
Initially, only the word with the highest coefficient (in absolute value) is selected. Then, at each iteration, the word that minimizes the following loss function is added:
$$\mathcal{L}(w)= \max_{v\in S} \langle w,v\rangle + \lambda c_w,$$
where $S$ is the set of already selected words and $c_w$ is the coefficient computed by the frozen weights method for the word $w$. $\lambda$ is a manually adjusted parameter.
 This step is iterated $M-1$ times, resulting in a total of M words and coefficients.
