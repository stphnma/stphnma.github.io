---
layout: default
title:  "The curious case of KO Hazard Rates in UFC fights"
date:   2017-11-29 
syntax_highlighter: rouge
---
<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>

# The curious case of hazard rates in UFC

I’ve recently scraped some UFC fights data, with the intention of applying some modeling on it in order to quantify fighter effectiveness. I’ve done some work before applying a Survival Analysis model on hockey data to quantify hockey player effectiveness, so I was curious to see what insights would come from applying the same model on MMA fighters.

I was playing around with the Kaplan Meier estimate of the survival curve, using KO as the failure event, when I noticed something interesting.

![Hazard Rate Duration]({{ "/assets/hazard_rate_duration.png" | absolute_url }})


There’s a large spike in the hazard rate around 300 seconds, which is right around when the 1st round ends.

What does this all mean? Well, for anyone still reading past this point, let’s provide some background first.



## Hold on, what’s Survival Analysis

**Survival analysis** is a field of statistics used to model the expected lifetime of an agent, before a “failure event” occurs. For example, a typical question in survival analysis would be “What’s the expected lifespan of prostate cancer patients?”, or “What’s the probability cancer patients survive up to 5 years?” etc.

The core of survival analysis lies within modelling the [survival function](https://en.wikipedia.org/wiki/Survival_function)  S(t), which defines the probability that the object of interest will survive up to time t.

$$S(t) = P[T \gt t]$$


The complement to the survival function is the **hazard rate** h(t), which is defined as the instantaneous probability that the event occurs at time t.

$$h(t) =  \lim \limits_{\delta \to 0} \frac{P[t \le T \le t+\delta  |  T \gt t]}{\delta}$$ 

The cool thing about survival analysis is that we can define the “failure event” as any arbitrary event. For example, financial institutions use survival analysis to model defaults on loans, where the default is the failure event. E-commerce companies use it to model customer churn, where canceling a membership is the failure event.

In this case, since we’re dealing with fight data, I’m using knockouts (KO/TKO) as the failure event.

The data I’ve scraped comprises of the UFC fights (up until last month), including the duration of the fight, who the fighters were, and the outcome (KO/TKO, submission, referee decision, etc.).

To model KO survival as function of time, we simply need the total duration of each fight, and label for each fight indicating whether or not the fight ended in a KO/TKO or not.

## Application

The lifelines library in python provides a nifty API for modeling survival functions. All you need to model the survival curve is a Pandas dataframe of fight duration and whether or not a KO/TKO occurred or not.

With the lifelines API, we can get a non-parametric estimate of the survival function through their `KaplanMeierFitter`.
{% highlight python %}

from lifelines import KaplanMeierFitter
kmf = KaplanMeierFitter()

kmf.fit(surv_df_3['fight_time'], event_observed=surv_df_3['KO'])
kmf.plot(title = "Probability of Surviving (a KO)", figsize = [15,7])
{% endhighlight %}

And we get something that looks like this:

![Survival Curve Duration]({{ "/assets/survival_curve_duration.png" | absolute_url }})


Interpretation of the curve is pretty straightforward. Each point represents the probabilty of surviving up to that point.

For example, there's a ~82% chance of not getting KO'd within the first round, and ~70% chance of not getting KO'd within the first 2 rounds. Ultimately, it looks like there's at least an 80% chance of not getting KO'd at all.

Notice the weird kink around 300 seconds? We’ll get back to that later.

Unfortunately, lifelines doesn’t provide a way to see the see the hazard function — however, we can approximate it by taking the shifted difference in ratio of the survival function.

{% highlight python %}
survival_func = kmf.survival_function_
shifted_survival_diff = (1 - survival_func/survival_func. shift(1))
hazard_empirical = shifted_survival_diff.div(survival_func.index.to_series(), axis=0)
{% endhighlight %}


Essentially, what we are doing is this:

$$ h(t_{i+1}) = \frac{\widehat{S}(t_{i+1}) - \widehat{S}(t_i)} {t_{i+1} - t_i}  \approx \frac{P[t_i \le T \le t_{i+1}  |  T \gt t_i]}{t_{i+1} - t_i} $$

Where $$\widehat{S}(t)$$ is the empirical survival function from the Kaplan Meier estimate.



Calculating and plotting the hazard rate now give us this:

![Hazard Rate Duration]({{ "/assets/hazard_rate_duration.png" | absolute_url }})

Some things pop out:
- the hazard rate of KO is highest in the beginning of the fight
- the hazard rate declines over time, but **spikes up again around 300 seconds** (remember the kink?)
- after that, it seems to stabilize and stay relatively constant

It makes sense that the hazard rate is highest in the beginning, as that's when fighters are still figuring each other out. In those early minutes, it's also possible for one fighter to get lucky with a swing and knock the other guy/gal out.

Over time, fighters get into their rhythym, and are able to react better to their opponenets, which explains the stabilization in hazard rates.

However, the increase at 300 seconds is pretty odd. Recall that the hazard rate is defined as the instantaneous rate of failure, or the probability of failing at time t (given that the agent has survived up until then). This means that right around 300 seconds, the probability of a KO jumps by more than 3x.

And if we look at the distribution of fight times, there’s also a spike around 300 seconds.

![Hazard Rate Duration]({{ "/assets/fight_times_hist.png" | absolute_url }})


So for some reason, the probability of a KO ~ 300 seconds is much higher than other times.

## What’s going on here?

To be honest, I’m not sure.

It’s possible that fighters more likely to let their guard right before the 1st round ends. Or alternatively, fighters become more aggressive in the end of the first round, hoping to score a KO.

Or potentially there's reason.

KO’s and TKO’s are judged by the sole discretion of the referee. The ref may call a KO/TKO if the fighter had been knocked unconscious, OR if the fighter had stopped defending himself/ herself. The second case is more ambiguous, and there have been complaints about early stoppage before. I wonder if a majority of these are early stoppages, and there’s some bias on the refs to call fights before the 1st round ends.

Conspiracy theories aside, what we can conclude is that fighters definitely need to keep their guard up in the seconds before the first round ends.