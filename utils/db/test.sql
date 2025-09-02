with episode_summary as (
    select
        episode,
        max(step) as final_step,
        sum(reward) as total_reward,
        mean(speed_ratio) as mean_speed_ratio,
    from data.experiments
    group by episode
),
last_steps as (
    select
        e.episode,
        e.step,
        e.new_target_distance as final_distance,
        e.result,
        e.is_train
    from data.experiments e
    join episode_summary s
      on e.episode = s.episode and e.step = s.final_step
),
first_steps as (
    select
        episode,
        target_distance as initial_distance
    from data.experiments
    where step = 0
)
select
    l.episode,
    l.step as final_step,
    f.initial_distance,
    l.final_distance,
    s.mean_speed_ratio,
    s.total_reward,
    l.result,
    l.is_train,
from last_steps l
join episode_summary s on l.episode = s.episode
left join first_steps f on l.episode = f.episode
where l.result <> 'fail'
  and l.is_train = false
order by l.final_distance;