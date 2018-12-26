# Important points for statistics of the corpus

This folder includes all the codes and results for processing [ARC corpus](http://data.allenai.org/arc/arc-corpus/).

According to Section 4.2 of [KG2 paper](https://arxiv.org/abs/1805.12393), before searching for the potential supports for each hypothesis, we need to filter the noisy sentences that:

1. contain negation words (e.g. not, except. etc)
2. contain unexpected characters
3. simply too long

However, since they did not provide the specific examples for each conditions, we used the following for our implementation:

1. negation words: no | not | none | neither | never | doesn't | does | isn't | aren't | wasn't | weren't | shouldn't | wouldn't | couldn't | won't | can't | cannot | don't | haven't | hadn't | except | hardly | scarcely | barely 
2. unexpected characters: we defined characters whose number of counts in the corpus less than 5000 as unexpected characters. For characters statistics for the corpus, please see common_characters.txt(characters in this file are considered as expected characters, and not removed from corpus),  common_characters_count.txt(characters appear more than 1000 times in the corpus), and unexpected_characters.txt (contains all the statistics for all characters except spaces, words and numbers). All of these files are under results folder.
3. simply too long: we define the sentence of length more than 60 as too long. To see the length statistics for sentences in the corpus, please see /results/statistics.txt. Why we choose 60: mean (17) + 3*standard variance 13 = 56, which is rounded to 60.