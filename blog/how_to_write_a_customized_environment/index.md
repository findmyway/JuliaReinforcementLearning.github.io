@def title = "How to write a customized environment?"
@def description = "The first step to apply algorithms in ReinforcementLearning.jl is to define the problem you want to solve in a recognizable way. Here we'll demonstrate how to write many different kinds of environments based on interfaces defined in [ReinforcementLearningBase.jl][]."
@def is_enable_toc = false
@def bibliography = "bibliography.bib"

@def front_matter = """
    {
        "authors": [
            {
                "author":"Jun Tian",
                "authorURL":"https://github.com/findmyway",
                "affiliation":"",
                "affiliationURL":""
            }
        ],
        "publishedDate":"$(now())"
    }"""

The most commonly used interfaces to describe reinforcement learning tasks is [OpenAI/Gym](https://gym.openai.com/). Inspired by it, we expand those interfaces a little to utilize the multiple-dispatch in Julia and to cover multi-agent environments.

## The minimal interfaces to implement

Many interfaces in [ReinforcementLearningBase.jl][] have a default implementation. So in most cases, you only need to implement the following functions to define a customized environment:

```julia
get_actions(env::YourEnv)
get_state(env::YourEnv)
get_reward(env::YourEnv)
get_terminal(env::YourEnv)
reset!(env::YourEnv)
(env::YourEnv)(action)
```

Here we use an example introduced in [Monte Carlo Tree Search: A Tutorial][] to demonstrate how to write a simple environment.

The game is defined like this: assume you have \$10 in your pocket, and you are faced with the following three choices:

1. Buy a PowerRich lottery ticket (win \$100M w.p. 0.01; nothing otherwise);
1. Buy a MegaHaul lottery ticket (win \$1M w.p. 0.05; nothing otherwise);
1. Do not buy a lottery ticket.

\dfig{page;LotteryEnv.png;Decision tree for lottery choices. Source: Figure 1 in [Monte Carlo Tree Search: A Tutorial][]}

First we define a concrete type named `LotteryEnv`, which is a subtype of `AbstractEnv`:

```julia:./lottery_env
using ReinforcementLearningBase

Base.@kwdef mutable struct LotteryEnv <: AbstractEnv
    reward::Union{Nothing, Int} = nothing
end
```

`LotteryEnv` has only one field named `reward`, by default it is initialized with `nothing`. Now let's implement the necessary interfaces:

```julia:./lottery_env
RLBase.get_actions(env::LotteryEnv) = (:PowerRich, :MegaHaul, nothing)
```

Here `RLBase` is just an alias for `ReinforcementLearningBase`. Based on the description above, we define a `Tuple` of 3 symbols as the possible actions for the `LotteryEnv`.

```julia:./lottery_env
RLBase.get_reward(env::LotteryEnv) = env.reward
RLBase.get_state(env::LotteryEnv) = !isnothing(env.reward)
RLBase.get_terminal(env::LotteryEnv) = !isnothing(env.reward)
RLBase.reset!(env::LotteryEnv) = env.reward = nothing
```

The lottery game is just a simple one-shot game. We take an action and then the reward is stored in the game. Then we can get the reward through `get_reward`. This simple game has only two states, `true` or `false`. If the `reward` field is `nothing` then the game is not terminated yet and we say the game is in state `false`, otherwise the game is terminated and the state is `true`. By `reset!`ing the game, we simply assign the reward with `nothing`, meaning that it's in the initial state.

The only left one to implement is the game logic:

```julia:./lottery_env
function (env::LotteryEnv)(action)
    if action == :PowerRich
        env.reward = rand() < 0.01 ? 100_000_000 : -10
    elseif action == :MegaHaul
        env.reward = rand() < 0.05 ? 1_000_000 : -10
    else
        env.reward = 0
    end
end
```

A simple way to check that your environment works is to apply the [`RandomPolicy`](https://juliareinforcementlearning.org/ReinforcementLearning.jl/latest/rl_base/#ReinforcementLearningBase.RandomPolicy) to the environment.

```julia:./lottery_env
env = LotteryEnv()
run(RandomPolicy(env), env)
```

## Traits of environments

If you run `LotteryEnv()` in the REPL, you'll get the following summary of the environment:

```julia:./show_lottery_env
# hideall
show(stdout, MIME"text/plain"(), LotteryEnv())  # hide
```

\output{./show_lottery_env}

The **Traits** section describes which categories the environment belongs to. Some of them are straightforward. The `NumAgentStyle` describes how many agents the environment to deal with. The `RewardStyle` describes when we get the reward. The default value is `StepReward()`, which means we can `get_reward(env)` after each step. But for some other games, for example [Tic-Tac-Toe](https://en.wikipedia.org/wiki/Tic-tac-toe) or [Go](https://en.wikipedia.org/wiki/Go_(game)), we only get reward at the end of game. For other traits, we'll explain them with examples one-by-one.

### Rock-Paper-Scissors

\dfig{body;https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Rock-paper-scissors.svg/440px-Rock-paper-scissors.svg.png;A chart showing how the three game elements interact. Source: [wiki](https://en.wikipedia.org/wiki/Rock_paper_scissors)}

Now let's see how to define one of the most traiditional multi-agent games: [Rock-Paper-Scissors](https://en.wikipedia.org/wiki/Rock_paper_scissors). First we need to define the struct `RockPaperScissorsEnv` which inherits `AbstractEnv`:

```julia
Base.@kwdef mutable struct RockPaperScissorsEnv <: AbstractEnv
    rewards::Union{Nothing, Tuple{Int, Int}} = nothing
end
```

The Rock-Paper-Scissors game has two players. We describe it with the `NumAgentStyle` like this:

```julia
RLBase.NumAgentStyle(::RockPaperScissorsEnv) = MultiAgent(2)
RLBase.get_players(::RockPaperScissorsEnv) = (:player1, :player2)
```

The possible actions for each player are exactly the same, so we have:

```julia
RLBase.get_actions(::RockPaperScissorsEnv) = (:Rock, :Paper, :Scissors)
```

And the Rock-Paper-Scissors game is a typical [simultaneous game](https://en.wikipedia.org/wiki/Simultaneous_game). Both players take actions at the same time:

```julia
RLBase.DynamicStyle(::RockPaperScissorsEnv) = SIMULTANEOUS
RLBase.get_reward(env::RockPaperScissorsEnv) = env.reward

function RLBase.get_reward(env::RockPaperScissorsEnv, player)
    if player == :player1
        env.reward[1]
    elseif player == :player2
        env.reward[2]
    else
        error("unknown player: $player")
    end
end

function (env::RockPaperScissorsEnv)(actions)
    if actions[1] == actions[2]
        env.reward = (0, 0)
    elseif actions[1] == :Rock && actions[2] == :Paper ||
        actions[1] == :Paper && actions[2] == :Scissors ||
        actions[1] == :Scissors && actions[2] == :Rock
        env.reward = (-1, 1)
    else
        env.reward = (1, -1)
    end
end
```

Similar to the `LotteryEnv`, Rock-Paper-Scissors is also a one-shot game.

```julia
RLBase.get_terminal(env::RockPaperScissorsEnv) = !isnothing(env.reward)
RLBase.get_state(env::RockPaperScissorsEnv) = !isnothing(env.reward)
```

### ActionStyle

```julia:./doc_of_ActionStyle
# hideall
print(@doc ActionStyle)
```

\textoutput{./doc_of_ActionStyle}

For environments of `FULL_ACTION_SET`, the following methods must be implemented:

- `get_legal_actions(env)`
- `get_legal_actions_mask(env)`

### DynamicStyle


```julia:./doc_of_DynamicStyle
# hideall
print(@doc DynamicStyle)
```

\textoutput{./doc_of_DynamicStyle}

For environment of `SIMULTANEOUS`, the actions in each step must be a collection, representing the joint actions from all players.

### UtilityStyle

```julia:./doc_of_UtilityStyle
# hideall
print(@doc UtilityStyle)
```

\textoutput{./doc_of_UtilityStyle}

### RewardStyle

```julia:./doc_of_RewardStyle
# hideall
print(@doc RewardStyle)
```

\textoutput{./doc_of_RewardStyle}

Some algorithms may use this trait for acceleration.

### ChanceStyle

```julia:./doc_of_ChanceStyle
# hideall
print(@doc ChanceStyle)
```

\textoutput{./doc_of_ChanceStyle}

Possible values are:

- `Deterministic`
- `Stochastic`
- `ExplicitStochastic`
- `SampledStochastic`

Some algorithms may only work on environments of `Deterministic` or `ExplicitStochastic`.

### InformationStyle

```julia:./doc_of_InformationStyle
# hideall
print(@doc InformationStyle)
```

\textoutput{./doc_of_InformationStyle}

### NumAgentStyle

```julia:./doc_of_NumAgentStyle
# hideall
print(@doc NumAgentStyle)
```

\textoutput{./doc_of_NumAgentStyle}

The `NumAgentStyle` trait is used to define the number of agents in an environment. Possible values are `SINGLE_AGENT` or `MultiAgent{N}()`. In multi-agent environments, a special case is `Two_Agent`, which is an alias of `MultiAgent{2}()`. For multi-agent environments, many functions need to accept another argument named `player` (for example `get_reward(env,player)`) to support getting information from the perspective of a specific player. Here's the list of these functions:

```julia:./list_of_multi_agent_methods
# hideall
for x in ReinforcementLearningBase.MULTI_AGENT_ENV_API
    println(x)
end
```

\output{./list_of_multi_agent_methods}

## Environment wrappers

Some useful environment wrappers are also provided in [ReinforcementLearningBase.jl][] to mimic OOP. For example, in the above `LotteryEnv`, actions are of type `Union{Symbol, Nothing}`. Some algorithms may require that the actions must be discrete integers. Then we can create a wrapped environment:

```julia
inner_env = LotteryEnv()
env = inner_env |> ActionTransformedEnv(a -> get_actions(inner_env)[a])
RLBase.get_actions(env::ActionTransformedEnv{<:LotteryEnv}) = 1:3
```

In some other cases, we may want to transform the state into integers. Similarly we can achieve this goal with the following code:

```julia
env = LotteryEnv() |> StateOverriddenEnv(s -> Int(s))
```

See the full list of other environment wrappers [here](https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/blob/master/src/implementations/environments.jl).

[ReinforcementLearningBase.jl]: https://github.com/JuliaReinforcementLearning/ReinforcementLearningBase.jl/blob/master/src/interface.jl
[Monte Carlo Tree Search: A Tutorial]: https://www.informs-sim.org/wsc18papers/includes/files/021.pdf