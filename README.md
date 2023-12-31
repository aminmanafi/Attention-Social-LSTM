# Attention-Social-LSTM

## Project details
**Authors:** Amin Manafi, Samaneh Hoseini

## abstract
In this paper, we propose a human trajectory prediction model that combines a Long Short-Term Memory (LSTM) network with an attention mechanism. To do that, we use attention scores to determine which parts of the input data the model should focus on when making predictions. Attention scores are calculated for each input feature, with a higher score indicating the greater significance of that feature in predicting the output. Initially, these scores are determined for the target human's position, velocity, and their neighboring individuals' positions and velocities. By using attention scores, our model can prioritize the most relevant information in the input data and make more accurate predictions.
We extract attention scores from our attention mechanism and integrate them into the trajectory prediction module to predict humans' future trajectories.
To achieve this, we introduce a new neural layer that processes attention scores after extracting them and concatenates them with positional information. We evaluate our approach on the publicly available ETH and UCY datasets and measure its performance using the final displacement error (FDE) and average displacement error (ADE) metrics. We show that our modified algorithm performs better than the Social LSTM in predicting the future trajectory of pedestrians in crowded spaces. Specifically, our model achieves an improvement of 6.2\% in ADE and 6.3\% in FDE compared to the Social LSTM results in the literature.

## setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2)
2. Run setup file: pip install -e .

## Train Attention scores network
python crowd_nav/train_crowd.py --policy sarl

## Train Attention Social LSTM
python crowd_naw/train.py