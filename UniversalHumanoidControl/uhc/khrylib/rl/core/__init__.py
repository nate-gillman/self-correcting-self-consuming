try:
    from uhc.khrylib.rl.core.common import *
    from uhc.khrylib.rl.core.critic import Value
    from uhc.khrylib.rl.core.distributions import DiagGaussian, Categorical
    from uhc.khrylib.rl.core.logger_rl import LoggerRL
    from uhc.khrylib.rl.core.policy import Policy
    from uhc.khrylib.rl.core.policy_disc import PolicyDiscrete
    from uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
    from uhc.khrylib.rl.core.trajbatch import TrajBatch
except:
    from UniversalHumanoidControl.uhc.khrylib.rl.core.common import *
    from UniversalHumanoidControl.uhc.khrylib.rl.core.critic import Value
    from UniversalHumanoidControl.uhc.khrylib.rl.core.distributions import DiagGaussian, Categorical
    from UniversalHumanoidControl.uhc.khrylib.rl.core.logger_rl import LoggerRL
    from UniversalHumanoidControl.uhc.khrylib.rl.core.policy import Policy
    from UniversalHumanoidControl.uhc.khrylib.rl.core.policy_disc import PolicyDiscrete
    from UniversalHumanoidControl.uhc.khrylib.rl.core.policy_gaussian import PolicyGaussian
    from UniversalHumanoidControl.uhc.khrylib.rl.core.trajbatch import TrajBatch