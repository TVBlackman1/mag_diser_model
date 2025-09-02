
select * from (select
    core_experiments.episode, step, new_target_distance as final_distance, result, reward_i, core_experiments.is_train
from data.experiments as core_experiments right join (
    select
        episode as episode_i, max(step) as step_i, sum(reward) as reward_i
    from data.experiments group by episode
) on core_experiments.episode=episode_i and core_experiments.step=step_i
) as core_experiments left join (
    select episode, target_distance as initial_distance from data.experiments where step=0
) as start_step_experiments on start_step_experiments.episode=core_experiments.episode where result <> 'fail' and is_train = false order by (final_distance);