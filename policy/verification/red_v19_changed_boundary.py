"""V19 clock-safe opening/search equivalence contracts; no Env or GPU model.

Real V19 opening, selector and material-board acceptance predicates execute.
Their V18 function bodies provide the behavior baseline. Public geometry/IK
boundaries are mocks, so passing is not physical placement or source renewal.
"""
import ast
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

WORK=Path(__file__).resolve().parents[1]
ROOT=WORK.parents[2]
RUN=WORK
sys.path.insert(0,str(WORK))
from puzzler import red_bimanual_goal as m
assert Path(m.__file__).resolve()==WORK/'puzzler/red_bimanual_goal.py'
BASELINE=ROOT/'runs/red_bimanual_regrasp_v18/full_ready/puzzler/red_bimanual_goal.py'
BASE_TREE=ast.parse(BASELINE.read_text())


def baseline_function(name):
    """Compile the actual old function body against the same boundary mocks."""
    node=next(n for n in BASE_TREE.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace=dict(m.__dict__)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(BASELINE),'exec'),namespace)
    return namespace[name]


class OpeningBoundary:
    def __init__(self,fault=None):
        self.state=dict(piece='red',arm=1,handoff={'candidate':{'bead':4}},
            parameters={'mode':'pyramid'},execution_tokens={},executed_placements=[])
        self.scene={'board_for_planning':{'pose':[.465,0.,.753,0.,0.,0.,1.],
            'uncertainty_m':.0002,'source_timestamp':202.},'revision':1}
        self.belief={'uncertainty_m':.006,'source':{'timestamp':200.}}
        self.q=np.arange(16,dtype=float)/100.;self.q[14:]=[.0121,-.0121]
        self.fault=fault;self.last_q=None;self.local_rows=[];self.external_results=[]
        self.geometry=SimpleNamespace(scratch=object())
        self.geometry.densify_q16_path=Mock(side_effect=lambda rows:np.linspace(rows[0],rows[1],35))
        self.geometry.screen_robot_material_path=Mock(side_effect=self.material)
        self.geometry.fk=Mock(side_effect=lambda q:setattr(self,'last_q',np.asarray(q).copy()))
        self.geometry.robot=SimpleNamespace(iks={'right':SimpleNamespace(ee_pose=lambda scratch:
            (np.asarray([self.last_q[8],0.,.9]),np.asarray([0.,0.,0.,1.])))})
        self.external=Mock(side_effect=self.external_result)
        self.local=Mock(side_effect=self.local_result)

    def body(self,g,q,arm,belief):
        if g is not self.geometry or arm!=1:raise AssertionError('Wrong public FK boundary')
        return np.asarray([.45+q[8],-.01,.82,0.,0.,0.,1.])

    def material(self,dense,scene,**kwargs):
        margin=m.FLOOR+scene['board_for_planning']['uncertainty_m']
        clearance=margin-1.e-7 if self.fault=='board_clearance' else margin
        groups={} if self.fault=='missing_board' else {'robot_board79':{'clearance_m':clearance}}
        rows=[] if self.fault=='missing_rows' else [dict(groups=deepcopy(groups)) for _ in dense]
        return dict(robot_material_safe=self.fault!='material',rows=rows)

    def external_result(self,state,scene,board,belief,body):
        result=dict(safe=self.fault!='external',minimum_unallowed_envelope_m=.0021,
            held_envelope_m=belief['uncertainty_m'],witness={'body':np.asarray(body).tolist(),
            'scene_revision':scene['revision'],'board':np.asarray(board).tolist()})
        self.external_results.append(result)
        return result

    def local_result(self,point,rotation,width,scene,board,belief):
        index=len(self.local_rows)%35
        safe=self.fault!=('local',index)
        result=dict(safe=safe,clearance_m=.0012 if safe else -.001,
            actual_width_m=width,held_envelope_m=belief['uncertainty_m'])
        self.local_rows.append(result)
        return result

    def install(self):
        stack=ExitStack()
        stack.enter_context(patch.object(m,'geometry',return_value=self.geometry))
        stack.enter_context(patch.object(m,'_body_at',side_effect=self.body))
        stack.enter_context(patch.object(m,'_terminal_external',self.external))
        stack.enter_context(patch.object(m,'_hand_guard',self.local))
        return stack


class ClockSafeOpeningContracts(unittest.TestCase):
    def opening(self,fault=None,old=False):
        rig=OpeningBoundary(fault)
        with rig.install():
            fn=baseline_function('_opening_at_q') if old else m._opening_at_q
            try:return rig,fn(rig.state,rig.scene,rig.belief,rig.q),None
            except m.c.PlanningRefusal as error:return rig,None,error

    def select(self,baseline,safe=lambda h:True,old=False):
        calls=[]
        def preview(ctx,state,scene,belief,height):
            calls.append(float(height))
            if not safe(height):raise m.c.PlanningRefusal('Original endpoint predicate refused')
            return {'requested_body_height_m':float(height),'original_gates_passed':True}
        state={'execution_tokens':{},'executed_placements':[]};scene={'original':'unchanged'}
        belief={'uncertainty_m':.006,'source':{'timestamp':200.}}
        before=deepcopy((state,scene,belief))
        with patch.object(m,'_preview_release_height',side_effect=preview):
            fn=baseline_function('_select_release_height') if old else m._select_release_height
            try:result=fn(object(),state,scene,belief,{'command_release_height_m':baseline});error=None
            except m.c.PlanningRefusal as refusal:result=None;error=refusal
        self.assertEqual((state,scene,belief),before)
        return result,calls,error

    def test_pass_matches_old_output_and_retains_35_material_35_local_external_once(self):
        rig,result,error=self.opening();old,expected,old_error=self.opening(old=True)
        self.assertIsNone(error);self.assertIsNone(old_error);self.assertEqual(result,expected)
        self.assertEqual(rig.external.call_count,1);self.assertEqual(old.external.call_count,35)
        self.assertEqual(rig.local.call_count,35);self.assertEqual(rig.geometry.fk.call_count,35)
        self.assertEqual(rig.geometry.screen_robot_material_path.call_count,1)
        material=rig.geometry.screen_robot_material_path.call_args
        self.assertEqual(len(material.args[0]),35);self.assertIs(material.args[1],rig.scene)
        self.assertEqual(material.kwargs['finger_target_contract'],{1:[4]})
        self.assertIsNone(material.kwargs['board_anchor_exit'])
        np.testing.assert_array_equal(material.kwargs['target_body_poses'],np.tile(result['body_before_open'],(35,1)))
        np.testing.assert_allclose([row['actual_width_m'] for row in result['local_hand_guards']],
            np.linspace(.0242,.090,35),atol=1.e-14)
        np.testing.assert_array_equal(result['q16_before_open'],rig.q)
        self.assertEqual(result['held_envelope_m'],.006);self.assertEqual(result['width_sweep_m'],[.0242,.090])

    def test_original_material_failures_refuse_with_external_local_unknown_and_zero_calls(self):
        for fault in ('material','board_clearance','missing_board','missing_rows'):
            with self.subTest(fault=fault):
                rig,result,error=self.opening(fault);old,expected,old_error=self.opening(fault,old=True)
                self.assertIsNone(result);self.assertIsNone(expected)
                self.assertIsNotNone(error);self.assertIsNotNone(old_error)
                self.assertEqual(rig.external.call_count,0);self.assertEqual(rig.local.call_count,0)
                self.assertEqual(rig.geometry.fk.call_count,0)
                payload=json.loads(str(error)[str(error).index('{'):])
                for name in ('external','local'):
                    self.assertEqual(payload[name]['status'],'UNKNOWN')
                    self.assertIn('Not evaluated',payload[name]['reason'])
                self.assertEqual(len(rig.geometry.screen_robot_material_path.call_args.args[0]),35)

    def test_original_external_failure_still_refuses_and_local_widths_are_all_checked(self):
        rig,result,error=self.opening('external');old,expected,old_error=self.opening('external',old=True)
        self.assertIsNone(result);self.assertIsNone(expected);self.assertIsNotNone(error);self.assertIsNotNone(old_error)
        self.assertEqual(rig.external.call_count,1);self.assertEqual(rig.local.call_count,35)
        self.assertEqual(old.external.call_count,35);self.assertEqual(old.local.call_count,35)
        payload=json.loads(str(error)[str(error).index('{'):]);self.assertFalse(payload['external']['safe'])

    def test_original_local_failure_at_first_middle_or_last_width_still_refuses(self):
        for index in (0,17,34):
            with self.subTest(index=index):
                rig,result,error=self.opening(('local',index))
                old,expected,old_error=self.opening(('local',index),old=True)
                self.assertIsNone(result);self.assertIsNone(expected)
                self.assertIsNotNone(error);self.assertIsNotNone(old_error)
                self.assertEqual(rig.local.call_count,35);self.assertEqual(rig.external.call_count,1)
                payload=json.loads(str(error)[str(error).index('{'):]);self.assertFalse(payload['local']['safe'])

    def test_external_result_is_deepcopied_into_every_jaw_row(self):
        rig,result,error=self.opening();self.assertIsNone(error)
        rows=result['held_external'];self.assertEqual(len(rows),35)
        self.assertTrue(all(row==rig.external_results[0] for row in rows))
        self.assertEqual(len({id(row) for row in rows}),35)
        self.assertEqual(len({id(row['witness']) for row in rows}),35)
        original=deepcopy(rig.external_results[0]);rows[0]['witness']['body'][0]=-99.
        self.assertEqual(rows[1],original);self.assertEqual(rig.external_results[0],original)

    def test_external_recomputed_for_each_call_scene_body_and_bound(self):
        rig=OpeningBoundary()
        with rig.install():
            first=m._opening_at_q(rig.state,rig.scene,rig.belief,rig.q)
            again=m._opening_at_q(rig.state,rig.scene,rig.belief,rig.q)
            self.assertEqual(rig.external.call_count,2);self.assertEqual(first,again)
            rig.scene['revision']=2;third=m._opening_at_q(rig.state,rig.scene,rig.belief,rig.q)
            self.assertEqual(third['held_external'][0]['witness']['scene_revision'],2)
            shifted=rig.q.copy();shifted[8]+=.003
            fourth=m._opening_at_q(rig.state,rig.scene,rig.belief,shifted)
            self.assertNotEqual(fourth['held_external'][0]['witness']['body'],third['held_external'][0]['witness']['body'])
            rig.belief['uncertainty_m']=.008
            fifth=m._opening_at_q(rig.state,rig.scene,rig.belief,shifted)
            self.assertEqual(fifth['held_external'][0]['held_envelope_m'],.008)
            self.assertEqual(rig.external.call_count,5);self.assertEqual(rig.local.call_count,175)
        self.assertEqual(first['held_external'][0]['witness']['scene_revision'],1)
        self.assertEqual(first['held_external'][0]['held_envelope_m'],.006)

    def test_lower_first_skips_nominal_after_lower_failure_with_same_selected_height(self):
        safe=lambda height:height>=.0068
        result,calls,error=self.select(.006174545519178914,safe)
        old,old_calls,old_error=self.select(.006174545519178914,safe,old=True)
        self.assertIsNone(error);self.assertIsNone(old_error);self.assertEqual(result[0],old[0])
        self.assertAlmostEqual(result[0],.0095)
        candidates=result[1]['candidates']
        expected=[row['lower_tracking_height_m'] for row in candidates]+[result[0]]
        np.testing.assert_allclose(calls,expected,atol=1.e-14);self.assertLess(len(calls),len(old_calls))
        self.assertEqual([r['qualified'] for r in candidates],[r['qualified'] for r in old[1]['candidates']])
        for row in candidates[:-1]:self.assertNotIn('command_opening',row)
        self.assertEqual(result[1]['selected'],old[1]['selected'])

    def test_lower_pass_never_bypasses_nominal_refusal(self):
        safe=lambda height:abs(height-.0095)>1.e-12
        result,calls,error=self.select(.0095,safe);old,_,old_error=self.select(.0095,safe,old=True)
        self.assertIsNone(error);self.assertIsNone(old_error);self.assertEqual(result[0],old[0])
        self.assertAlmostEqual(result[0],.010)
        np.testing.assert_allclose(calls,[.007,.0095,.0075,.010],atol=1.e-14)
        self.assertFalse(result[1]['candidates'][0]['qualified'])
        self.assertIn('lower_tracking_opening',result[1]['candidates'][0])
        self.assertNotIn('command_opening',result[1]['candidates'][0])

    def test_safe_nongrid_clamp_and_maximum_preserve_old_height_and_both_endpoints(self):
        for baseline,height in ((.00975,.00975),(.001,.005),(.011,.011)):
            with self.subTest(baseline=baseline):
                result,calls,error=self.select(baseline);old,_,old_error=self.select(baseline,old=True)
                self.assertIsNone(error);self.assertIsNone(old_error);self.assertEqual(result,old)
                self.assertEqual(result[0],height);np.testing.assert_allclose(calls,[height-.0025,height])
                self.assertEqual(result[1]['maximum_command_height_m'],.011)
                self.assertEqual(result[1]['tracking_vertical_shortfall_m'],.0025)
                self.assertTrue(result[1]['public_goal_unchanged']);self.assertTrue(result[1]['no_motor_in_selection'])
        result,calls,error=self.select(.0115);old,old_calls,old_error=self.select(.0115,old=True)
        self.assertIsNone(result);self.assertIsNone(old);self.assertIsNotNone(error);self.assertIsNotNone(old_error)
        self.assertEqual(calls,[]);self.assertEqual(old_calls,[])

    def test_all_rejected_search_is_finite_with_same_candidate_bounds(self):
        result,calls,error=self.select(.005,lambda height:False)
        old,old_calls,old_error=self.select(.005,lambda height:False,old=True)
        self.assertIsNone(result);self.assertIsNone(old);self.assertIsNotNone(error);self.assertIsNotNone(old_error)
        np.testing.assert_allclose(calls,np.arange(10,23)/2000.-.0025,atol=1.e-14)
        np.testing.assert_allclose(old_calls,np.arange(10,23)/2000.,atol=1.e-14)

    def test_unexpected_geometry_error_is_not_suppressed(self):
        for old in (False,True):
            with self.subTest(old=old):
                rig=OpeningBoundary()
                with rig.install():
                    rig.geometry.screen_robot_material_path.side_effect=RuntimeError('geometry unavailable')
                    fn=baseline_function('_opening_at_q') if old else m._opening_at_q
                    with self.assertRaisesRegex(RuntimeError,'geometry unavailable'):
                        fn(rig.state,rig.scene,rig.belief,rig.q)
                self.assertEqual(rig.external.call_count,0);self.assertEqual(rig.local.call_count,0)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ClockSafeOpeningContracts)
    checks=[test.id().rsplit('.',1)[-1] for test in suite]
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report=dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),checks=checks,
        module=str(Path(m.__file__).resolve()),baseline=str(BASELINE),scope=__doc__,
        controller_modified=False,environment_constructed=False,GPU_model_created=False,
        failed_checks=[dict(test=test.id(),traceback=trace) for test,trace in [*result.failures,*result.errors]])
    (WORK/'verification/red_v19_changed_boundary.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
