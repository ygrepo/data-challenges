# CostlyConversion
## Please limit yourself to 4 hours time!
### Place your submissions to the Submissions sub-folder with the naming convention: lname_fname.


## Goal

You sit inside the product team at Symantec, which sells a cyber security software for $39 across a variety of devices. Revenue has been flat for some time, so the VP of Product would like to experiment with the effect of increasing the price. Your team designed an experiment to measure the effect of doing so. In the experiment, 66% of the users have seen the old price ($39), while a random sample of 33% users were shown a higher price ($59). 

The experiment has been running for some time and the VP of Product is interested in understanding how it went. She would like to learn key insights about what drives conversion rate. She'd like a recommendation of what price to sell the software at, and for you to quantify the cost of the experiment and whether you could have done it in shorter time. Would you have designed the experiment differently, why or why not?

## Data
We have two tables downloadable by clicking here. The two tables are:

### "test_results" - data about the test
Columns:

user_id : the Id of the user. Can be joined to user_id in user_table

timestamp : the date and time when the user hit for the first time company XYZ webpage. It is in user local time

source : marketing channel that led to the user coming to the site. It can be: ads-["google", "facebook", "bing", "yahoo", "other"]. That is, user coming from google ads, yahoo ads, etc. seo - ["google", "facebook", "bing", "yahoo", "other"]. That is, user coming from google search, yahoo, facebook, etc.

friend_referral : user coming from a referral link of another user

direct_traffic: user coming by directly typing the address of the site on the browser

device : user device. Can be mobile or web

operative_system : user operative system. Can be: "windows", "linux", "mac" for web, and "android", "iOS" for mobile. "Other" if it is none of the above

test: whether the user was in the test (i.e. 1 -> higher price) or in control (0 -> old, lower price)

price : the price the user sees. It should match test

converted : whether the user converted (i.e. 1 -> bought the software) or not (0 -> left the site without buying it).

### "user_table" - Information about the user
Columns:
user_id : the Id of the user. Can be joined to user_id in test_results table

city : the city where the user is located. Comes from the user ip address

country : in which country the city is located

lat : city latitude - should match user city

long : city longitude - should match user city

