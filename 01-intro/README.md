# Introduction to Machine Learning
## Motivating Use Case
### The Challenge
Image we own a web platform for buying and selling used cars. Users upload details and pictures of vehicles
they want to sell, and need to set a price. This last step is often tricky: a user will want to set a price
that can attract buyers while maximizing the return on the sale.

The value of a used car is dependent on a wide range of factors, from mileage to seating and fuel efficiency (among others).
The ability to assess all these factors and determine a selling price that will satisfy both seller and buyer is often
the product of experience. We would expect a car salesman to draw on the experience of previous deals to recognize
how different characteristics of a car impact its value, and be able to assign a price that would be considered 'fair'.

### How Machine Learning helps
In the absence of an expert on our virtual self-service platform, we rely on a machine learning model to guide pricing decisions.
Much like the veteran car dealer, machine learning extracts patterns from historical data to build 'expertise' that it
later applies to new information. If we, as developers, can gather enough information about past car sales that a model
can learn from, we can equip the platform with a virtual car pricing expert.

For a machine learning model to be accurate, we must give it the type of information it needs to complete the task at hand.
In our case, these would be the set of factors that drive the price of a car. A sensible list would include mileage,
number of seats, make and model, among others.
In machine learning jargon, the set of variables that a model uses for prediction is referred to as 'features'.
The predicted value (in our example, the car price) is identified as the 'target'.

In summary, we want a machine learning model that utilizes various features (what we know) to predict
an unknown target (what we want to know). This is achieved through a process of extracting patterns from data,
which the model can then apply to new observations to make predictions.
For example, this approach can be used to estimate the value of a used car or to detect diseases in medical scans.

## Machine Learning vs Rule-based Systems