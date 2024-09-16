use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::evaluation_gpu::ProveExpression;
use halo2_proofs::plonk::evaluation_gpu::ProveExpressionUnit;
use halo2_proofs::plonk::Any;
use halo2_proofs::plonk::Expression;
use halo2_proofs::plonk::ProvingKey;
use std::collections::BTreeMap;
use std::collections::HashSet;

pub(crate) fn analyze_expr_tree<F: FieldExt>(
    expr: &ProveExpression<F>,
    k: usize,
) -> Vec<Vec<(BTreeMap<ProveExpressionUnit, u32>, BTreeMap<u32, F>)>> {
    let tree = expr.clone().flatten();
    let tree = tree
        .into_iter()
        .map(|(us, v)| {
            let mut map = BTreeMap::new();
            for mut u in us {
                if let Some(c) = map.get_mut(&mut u) {
                    *c = *c + 1;
                } else {
                    map.insert(u.clone(), 1);
                }
            }
            (map, v.clone())
        })
        .collect::<Vec<_, _>>();

    let limit = if k < 23 { 26 } else { 10 };
    let mut v = HashSet::new();

    let mut expr_group = vec![];
    let mut expr_groups = vec![];
    for (_, (units, coeff)) in tree.iter().enumerate() {
        let mut v_new = v.clone();
        let mut v_new_clean = HashSet::new();
        let mut muls_new = 0;
        for (unit, exp) in units {
            v_new.insert(unit.get_group());
            v_new_clean.insert(unit.get_group());
            muls_new += exp;
        }

        if v_new.len() > limit {
            v = v_new_clean;

            expr_groups.push(expr_group);
            expr_group = vec![(units.clone(), coeff.clone())];
        } else {
            v = v_new;
            expr_group.push((units.clone(), coeff.clone()));
        }
    }

    expr_groups.push(expr_group);
    expr_groups
}

fn _collect_involved_advices<F: FieldExt>(exprs: &[Expression<F>], units: &mut HashSet<usize>) {
    for expr in exprs {
        for (k, _) in ProveExpression::from_expr(expr).flatten() {
            for unit in k {
                match unit {
                    ProveExpressionUnit::Fixed { .. } => {}
                    ProveExpressionUnit::Advice { column_index, .. } => {
                        units.insert(column_index);
                    }
                    ProveExpressionUnit::Instance { .. } => {}
                }
            }
        }
    }
}

pub(crate) fn _analyze_involved_advices<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> (
    HashSet<usize>,
    HashSet<usize>,
    HashSet<usize>,
    HashSet<usize>,
    HashSet<usize>,
) {
    let mut tuple_lookups_involved_units = HashSet::new();
    let mut lookups_involved_units = HashSet::new();
    let mut permutation_involved_units = HashSet::new();
    let mut shuffle_involved_units = HashSet::new();

    let mut uninvolved_units = HashSet::new();
    for i in 0..pk.vk.cs.num_advice_columns {
        uninvolved_units.insert(i);
    }

    for lookup in &pk.vk.cs.lookups {
        if lookup.input_expressions.len() > 1 {
            _collect_involved_advices(
                &lookup.input_expressions[..],
                &mut tuple_lookups_involved_units,
            );
            _collect_involved_advices(
                &lookup.table_expressions[..],
                &mut tuple_lookups_involved_units,
            );
        } else {
            _collect_involved_advices(&lookup.input_expressions[..], &mut lookups_involved_units);
            _collect_involved_advices(&lookup.table_expressions[..], &mut lookups_involved_units);
        }
    }

    for c in &pk.vk.cs.permutation.columns {
        match c.column_type() {
            Any::Advice => {
                permutation_involved_units.insert(c.index());
            }
            _ => {}
        }
    }

    for shuffle in &pk.vk.cs.shuffles.0 {
        _collect_involved_advices(&shuffle.input_expressions[..], &mut shuffle_involved_units);
        _collect_involved_advices(
            &shuffle.shuffle_expressions[..],
            &mut shuffle_involved_units,
        );
    }

    for i in tuple_lookups_involved_units.iter() {
        lookups_involved_units.insert(*i);
    }

    for i in lookups_involved_units.iter() {
        uninvolved_units.remove(i);
    }

    for i in permutation_involved_units.iter() {
        uninvolved_units.remove(i);
    }

    for i in shuffle_involved_units.iter() {
        uninvolved_units.remove(i);
    }

    return (
        tuple_lookups_involved_units,
        lookups_involved_units,
        permutation_involved_units,
        shuffle_involved_units,
        uninvolved_units,
    );
}
