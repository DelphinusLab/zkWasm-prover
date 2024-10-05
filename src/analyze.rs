use halo2_proofs::arithmetic::CurveAffine;
use halo2_proofs::arithmetic::FieldExt;
use halo2_proofs::plonk::evaluation_gpu::ProveExpression;
use halo2_proofs::plonk::evaluation_gpu::ProveExpressionUnit;
use halo2_proofs::plonk::Any;
use halo2_proofs::plonk::Expression;
use halo2_proofs::plonk::ProvingKey;
use std::collections::BTreeMap;
use std::collections::HashSet;

use crate::expr::is_expression_pure_unit;

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

pub(crate) fn lookup_classify<'a, 'b, C: CurveAffine, T>(
    pk: &'b ProvingKey<C>,
    lookups_buf: Vec<T>,
) -> [Vec<(usize, T)>; 3] {
    let mut single_unit_lookups = vec![];
    let mut single_comp_lookups = vec![];
    let mut tuple_lookups = vec![];

    pk.vk
        .cs
        .lookups
        .iter()
        .zip(lookups_buf.into_iter())
        .enumerate()
        .for_each(|(i, (lookup, buf))| {
            let is_single =
                lookup.input_expressions.len() == 1 && lookup.table_expressions.len() == 1;

            if is_single {
                let is_unit = is_expression_pure_unit(&lookup.input_expressions[0])
                    && is_expression_pure_unit(&lookup.table_expressions[0]);
                if is_unit {
                    single_unit_lookups.push((i, buf));
                } else {
                    single_comp_lookups.push((i, buf));
                }
            } else {
                tuple_lookups.push((i, buf))
            }
        });

    return [single_unit_lookups, single_comp_lookups, tuple_lookups];
}

fn collect_involved_advices<F: FieldExt>(exprs: &[Expression<F>], units: &mut HashSet<usize>) {
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

pub(crate) fn analyze_involved_advices<C: CurveAffine>(
    pk: &ProvingKey<C>,
) -> (
    HashSet<usize>,
    HashSet<usize>,
    HashSet<usize>,
    HashSet<usize>,
    HashSet<usize>,
) {
    let mut uninvolved_units = HashSet::new();
    let mut uninvolved_units_after_single_lookup = HashSet::new();
    let mut uninvolved_units_after_tuple_lookup = HashSet::new();
    let mut uninvolved_units_after_permutation = HashSet::new();
    let mut uninvolved_units_after_shuffle = HashSet::new();

    for i in 0..pk.vk.cs.num_advice_columns {
        uninvolved_units.insert(i);
    }

    for lookup in &pk.vk.cs.lookups {
        if lookup.input_expressions.len() > 1 {
            collect_involved_advices(
                &lookup.input_expressions[..],
                &mut uninvolved_units_after_tuple_lookup,
            );
            collect_involved_advices(
                &lookup.table_expressions[..],
                &mut uninvolved_units_after_tuple_lookup,
            );
        } else {
            collect_involved_advices(
                &lookup.input_expressions[..],
                &mut uninvolved_units_after_single_lookup,
            );
            collect_involved_advices(
                &lookup.table_expressions[..],
                &mut uninvolved_units_after_single_lookup,
            );
        }
    }

    for c in &pk.vk.cs.permutation.columns {
        match c.column_type() {
            Any::Advice => {
                uninvolved_units_after_permutation.insert(c.index());
            }
            _ => {}
        }
    }

    for shuffle in &pk.vk.cs.shuffles.0 {
        collect_involved_advices(
            &shuffle.input_expressions[..],
            &mut uninvolved_units_after_shuffle,
        );
        collect_involved_advices(
            &shuffle.shuffle_expressions[..],
            &mut uninvolved_units_after_shuffle,
        );
    }

    for i in uninvolved_units_after_shuffle.iter() {
        uninvolved_units.remove(i);
        uninvolved_units_after_single_lookup.remove(i);
        uninvolved_units_after_tuple_lookup.remove(i);
        uninvolved_units_after_permutation.remove(i);
    }

    for i in uninvolved_units_after_permutation.iter() {
        uninvolved_units.remove(i);
        uninvolved_units_after_single_lookup.remove(i);
        uninvolved_units_after_tuple_lookup.remove(i);
    }

    for i in uninvolved_units_after_tuple_lookup.iter() {
        uninvolved_units.remove(i);
        uninvolved_units_after_single_lookup.remove(i);
    }

    for i in uninvolved_units_after_single_lookup.iter() {
        uninvolved_units.remove(i);
    }

    return (
        uninvolved_units,
        uninvolved_units_after_single_lookup,
        uninvolved_units_after_tuple_lookup,
        uninvolved_units_after_permutation,
        uninvolved_units_after_shuffle,
    );
}
