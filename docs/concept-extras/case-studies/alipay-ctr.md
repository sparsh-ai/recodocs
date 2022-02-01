# Alipay CTR

An online A/B testing was conducted in the production environment of Alipay for 10 days. The candidate items recommended to users include cash reward, coupons, prizes and member credits. The goal is to increase the CTR of the candidate items while constraining the total cost due to limited budget. The recommendation system serves at the scale of tens of millions of users in real traffic, hence the traffic is very expensive in the business view.

![Online A/B testing results of MIAN and DCN.](/img/content-concepts-case-studies-raw-case-studies-alipay-ctr-untitled.png)

Online A/B testing results of MIAN and DCN.

MIAN model brings a 0.41% gain in CTR while a 0.27% drop in cost compared to the best baseline method DCN in a statistically significant level which contributes a considerable business revenue growth. Note that, on a large scale commercial platform, an increase like 0.1% in CTR value can bring huge benefits. Besides, as shown in above figure, the CTR gain of the model is consistent during the 10-days online experiment, which demonstrates the effectiveness and stability of this model.