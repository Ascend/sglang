def derive_parallel_config(model_config, device_info):
    # Step 1: Determine minimum TP (based on hidden_size)
    hidden_size = model_config.hidden_size
    min_tp = max(1, hidden_size // 2048)  # 1 TP recommended per 2048 hidden_dim
    
    # Step 2: MoE models need EP
    if model_config.is_moe:
        n_experts = model_config.n_routed_experts
        # EP must divide the number of experts
        valid_ep_values = [e for e in range(1, n_experts + 1) 
                          if n_experts % e == 0]
        # TP must be divisible by EP or EP must be divisible by TP
        valid_ep_values = [e for e in valid_ep_values 
                          if min_tp % e == 0 or e % min_tp == 0]
        ep_size = max(valid_ep_values) if valid_ep_values else 1
    else:
        ep_size = 1
    
    # Step 3: Determine TP (must satisfy TP % EP == 0)
    if ep_size > min_tp:
        tp_size = ep_size
    else:
        tp_size = max(min_tp, ep_size)
        while tp_size % ep_size != 0:
            tp_size += 1
    
    # Step 4: Check device count constraints
    if tp_size > device_info.device_count:
        # Downgrade strategy: reduce TP
        tp_size = device_info.device_count
        ep_size = 1  # MoE downgraded to pure TP
    
    # Step 5: PP only used for extra large models
    pp_size = 1
    
    return tp_size, ep_size, pp_size