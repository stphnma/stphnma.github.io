---
layout: default
title:  "The curious case of hazard rates in UFC"
date:   2017-11-29 
syntax_highlighter: rouge
---

# The curious case of hazard rates in UFC

I’ve recently scraped some UFC fights data, with the intention of applying some modeling on it in order to quantify fighter effectiveness. I’ve done some work before applying a Survival Analysis model on hockey data to quantify hockey player effectiveness, so I was curious to see what insights would come from applying the same model on MMA fighters.

I was playing around with the Kaplan Meier estimate of the survival curve, using KO as the failure event, when I noticed something interesting.

There’s a huge spike in the hazard rate around 300 seconds, which is right around when the 1st round ends .

What does this all mean? Well, for anyone still reading past this point, let’s provide some background first.


## Hold on, what’s Survival Analysis

Survival analysis is a field of statistics used to model the expected lifetime of an agent, before a “failure event” occurs. For example, a typical question in survival analysis would be “What’s the expected lifespan of prostate cancer patients?”, or “What’s the probability cancer patients survive up to 5 years?” etc.

The core of survival analysis lies within modelling the [survival function](https://en.wikipedia.org/wiki/Survival_function)  S(t), which defines the probability that the object of interest will survive up to time t.

The complement to the survival function is the hazard rate h(t), which is defined as the instantaneous probability that the event occurs at time t.

The cool thing about survival analysis is that we can define the “failure event” as any arbitrary event. For example, financial institutions use survival analysis to model defaults on loans, where the default is the failure event. E-commerce companies use it to model customer churn, where canceling a membership is the failure event.

In this case, since we’re dealing with fight data, I’m using knockouts (KO/TKO) as the failure event.

## Application

The `lifelines` library in python provides a nifty API for modeling survival functions. All you need to model the survival curve is input the observed lifetime for each event (which in this case is the number of seconds before KO / or fight ending), and
With the lifelines API, we can get a non-parametric estimate of the survival function through their `KaplanMeierFitter`  , which calculates

for each point in time t.

And we get something that looks like this:

![Survival Curve Duration]({{ "/assets/survival_curve_duration.png" | absolute_url }})


Interpretation of the curve is pretty straightforward. Each point represents the probabilty of surviving up to that point.

For example, there's a ~82% chance of not getting KO'd within the first round, and ~70% chance of not getting KO'd within the first 2 rounds. Ultimately, it looks like there's at least an 80% chance of not getting KO'd at all.

Notice the weird kink around 300 seconds? We’ll get back to that later.

Unfortunately, lifelines doesn’t provide a way to see the see the hazard function — however, we can approximate it by taking the shifted difference in ratio of the survival function.

Essentially, what we are approximating is

 1 - S(t + dt) / S(t) ~~ P[ T < t+dt | T > t ] , which is probability of failure (KO) at time t+dt, which is the definition of the hazard rate.

Calculating and plotting the hazard rate now give us this:

![Hazard Rate Duration]({{ "/assets/hazard_rate_duration.png" | absolute_url }})


Now, the kink from before shows up in a big way. Recall that the hazard rate is defined as the instantaneous rate of failure, or the probability of failing at time t (given that the agent has survived up until then).

That means that right around 300 seconds, the probability of a KO jumps by more than 2x.

If we look at the distribution of fight times, there’s also a spike around 300 seconds.

![Hazard Rate Duration]({{ "/assets/hazard_rate_duration.png" | absolute_url }})


So for some reason, the probability of a KO ~ 300 seconds is much higher than other times.

## What’s going on here?

To be honest, I’m not sure.

It’s possible that fighters more likely to let their guard right before the 1st round ends. Or alternatively, fighters become more aggressive in the end of the first round, hoping to score a KO.

Or potentially there's reason.

KO’s and TKO’s are judged by the sole discretion of the referee. The ref may call a KO/TKO if the fighter had been knocked unconscious, OR if the fighter had stopped defending himself/ herself. The second case is more ambiguous, and there have been complaints about early stoppage before. I wonder if a majority of these are early stoppages, and there’s some bias on the refs to call fights before the 1st round ends.

Conspiracy theories aside, what we can conclude is that fighters definitely need to keep their guard up in the seconds before the first round ends.