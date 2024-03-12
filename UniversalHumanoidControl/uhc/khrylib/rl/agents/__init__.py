try:
    from uhc.khrylib.rl.agents.agent_pg import AgentPG
    from uhc.khrylib.rl.agents.agent_ppo import AgentPPO
    from uhc.khrylib.rl.agents.agent_trpo import AgentTRPO
except:
    from UniversalHumanoidControl.uhc.khrylib.rl.agents.agent_pg import AgentPG
    from UniversalHumanoidControl.uhc.khrylib.rl.agents.agent_ppo import AgentPPO
    from UniversalHumanoidControl.uhc.khrylib.rl.agents.agent_trpo import AgentTRPO
