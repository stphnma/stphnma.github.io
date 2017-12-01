---
layout: default
title:  "Likelihood of getting knocked out? A survival analysis of UFC fights"
date:   2017-10-22 16:16:01 -0600
---


The core of survival analysis lies within modelling the [survival function](https://en.wikipedia.org/wiki/Survival_function)  S(t), which defines the probability that a object of interest will survive up to time t. In this case, since we're modelling knockouts, our survival functions is the probability that a fighter won't get KO'd before a certain point.

Let see what it looks like:

{% highlight python %}

from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

kmf.fit(surv_df_3['fight_time'], event_observed=surv_df_3['KO'])
kmf.plot(title = "Probability of Surviving (a KO)", figsize = [15,7])
{% endhighlight %}


![Survival Curve Duration]({{ "/assets/survival_curve_duration.png" | absolute_url }})

Notice the weird kink at ~ 300 seconds? We'll to that shortly.

Interpretation of the curve is pretty straightforward. Each point represents the probabilty of surviving up to that point. 

For example, there's a ~82% chance of not getting KO'd within the first round, and ~70% chance of not getting KO'd within the first 2 rounds. Ultimately, it looks like there's at least an 80% chance of not getting KO'd at all, for a 3 round fight.


The other complement to the survival function is the hazard rate h(t), which is defined as the instantaneous probability that the event occurs at time t. Lifelines doesn't have a way of accessing the hazard function, but we can approximate it by taking the ratio of survival probability at time t against time t-1, which the probability of surving till t given surviving up to t-1, and subtracting that from one.

{% highlight python %}

hazard = (1 - kmf.survival_function_/kmf.survival_function_. shift(1))
ax = hazard.plot(title = "Hazard Rate (apprx.)",figsize = [15,7])
{% endhighlight %}

![Hazard Rate Duration]({{ "/assets/hazard_rate_duration.png" | absolute_url }})

It's not pretty, but the kink at ~ 300 seconds really stands out now. 300 seconds also happens to be just how long a single round is in the UFC, which means that somehow in the few seconds before the first round ends, the probability of getting knocked out jumps by 2X. 

This is pretty strange. Maybe there's often a flurry of fisticuffs right before the bell rings in round 1, which results in the KO. Or, possibly something else is going one.

A KO in the UFC is defined as either a knockout, where someone is knocked unconscious or technical knockout, which is a stoppage by the referee when he/she determines that the fighter cannot continue. A TKO is determined by the referee when he/she sees that the fighter has stopped defending himself/herself and is not fighting back. I wonder if there might be some kind of bias on their side to stopping a fight early in the first round. There's definitely an economic incentive for the UFC to have as many first round knockouts as possible.

## Survival based on other factors

Conspiracy theories aside aside, we can use other variables as our time variable. Let's look at survival as a function of head strikes received, instead of time (since those are the things that should directly lead to KO's).

![Hazard Rate Duration]({{ "/assets/survival_curve_head_strikes.png" | absolute_url }})


Interestingly enough, past 50 head strikes, the survival curve starts to plateau. I'm guessing there's a few factors at play such as:
- after a certain point, there's a diminishing marginal power for each additional strike, due to fatigue
- there are some fighters who are excellent strikers, but don't have "knockout" power -- we're probably seeing a lot of these guys in the high head strike volume range.

We can also calculate the survival function based on number of strikes on other body parts (namely the body and leg). Let's plot them on top of each other to compare effectiveness.

![Hazard Rate Duration]({{ "/assets/survival_curve_all_strikes.png" | absolute_url }})


This is actually very interesting! The plot seems to suggest that body and leg strikes are much more effective than head strikes at increasing a fighter's probability of getting KO'ed. 