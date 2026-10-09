from infra.stacks import naming as n


def test_only_beta_has_dedicated_read_only_strategy_lambda(assembly):
    for env in ('beta','gamma','prod'):
        name = f'finplan-{env}-financemodel-control'
        functions = assembly.resources(name,'AWS::Lambda::Function')
        found = [f for f in functions.values() if f['Properties']['FunctionName'] == n.function_name(env,n.STRATEGY_INFERENCE)]
        if env != 'beta':
            assert found == []
            continue
        assert len(found) == 1
        props = found[0]['Properties']
        assert props['Handler'] == 'finplan_model.serving.handler'
        assert props['Timeout'] == 270 and props['Architectures'] == ['arm64']
        assert 'FINPLAN_RUNS_TABLE' not in props['Environment']['Variables']
        assert 'StrategyFunctionRef' in assembly.stack(name)['Outputs']
