# BlueYp713_UMD_2022

## Made by Chanow, Chung, Farmer

**Summary:**

  Our team started by answering the four questions that were asked of the Armed Conflict Location and Events dataset. We found that the overall number of protests has decreased since June 2020, but there were local spikes that could be associated with events going on around the country. We looked at the most common types of events and saw that protests were overwhelmingly more common than any other type of event. We excluded protests and saw that the number of riots was originally greater than strategic developments, but that the number of strategic developments then increased after that. Then we looked at the most prominent event type of event in each county. We saw that almost every county had the most prominent type of events being protests. We excluded protests but did not see any meaningful conclusions from that map. Finally, we looked at how the number of types of events changed in the three most populated counties in the United States. We found that the trend in number of events in these counties largely followed the overall trend of events in the U.S.
  
  
  We decided to take a deeper look at the data by analyzing the most common actor in the dataset which was BLM: Black Lives Matter. We found that BLM protests over time were very similar to the general graphs we had found. There was a huge spike in June 2020 in response to the death of George Floyd. We also found that Los Angeles had the greatest number of BLM protests, so we decided to look more into what was going on in LA. We found many news articles about violent protests, looting, vandalism, and fire that occured during BLM protests. However, we analyzed the sub_event_types for these protests and a very large number of the events were categorized as peaceful protests. Thus, I wanted to disprove the myth that many BLM protests escalate to violence. News articles may only emphasize how BLM protests are violent and that is what most people could believe, but that is not true. 
  
  
  Afterwards, we decided to see if we could build a predictive model that could categorize the sub_event_type of a future event. We created a dataset that contained the location, month, actor, and sub_event_type of each row of the original dataset. We used this data for our predictive models. The first model we explored was K-Nearest Neighbors Classifier. This model did not perform very well because no matter what diverse input you give the model it would give back a sub_event_type of peaceful protest. For example, if I wanted to predict an event in DC during the month of December by a labor group, the predicted event would be a peaceful protest. 
  
  
  The second classification system we attempted was a low-rank matrix completion model. The rank of the data was determined to be 15 through trial and error. The testing loss of this matrix completion system (Gradient Descent) did bottom out at around 200 epochs with a learning rate of 0.01, however the change in the loss from the beginning to end of training was so low that we determined this system was not effective. 
  
  
  The final classification system we created was a tensorflow dense neural network. This network was 3 layers and had 16 classification labels. The adam optimizer was used. This model had a linear decreasing loss on training and a decreasing loss on testing data as well. If this system was formatted as binary instead of 16 labels it may have actually been very effective. 


