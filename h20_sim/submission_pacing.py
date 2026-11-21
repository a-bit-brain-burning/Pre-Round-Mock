from collections import Counter
from math import floor


class SubmissionPacer:
    def __init__(self,limit=100,task_count=3,hours=8):
        self.limit=limit
        self.task_count=task_count
        self.duration_ms=hours*3600000
        self.start_ms=None
        self.used=Counter()
        self.seen=0

    def decide(self, problem, repeat, incumbent, score, action, start_ms, probe_margin, small_signal=False,
               gain_signal=False,score_high=None):
        if repeat:
            return "SKIP_REPEAT"
        if self.start_ms is None:
            self.start_ms=start_ms
        self.seen+=1
        reserved=self.task_count-len(set(self.used)|{problem})
        spent=self.used.total()
        if spent+reserved>=self.limit:
            return "SKIP_BUDGET"
        if action=="SKIP_CAP":
            return action
        progress=max(min(1,(start_ms-self.start_ms)/self.duration_ms),self.seen/(self.seen+self.limit))
        regular=floor(self.limit*(0.25+0.75*progress))
        # 新题、缺少参照或接近纪录的版本可提前使用部分为后续保留的名额。
        under_share=self.used[problem]*self.task_count<spent
        credible=score_high is not None and score is not None and (score+score_high)/2>incumbent
        near=(small_signal or score is None or score>=incumbent-max(10,probe_margin)
              or (under_share and (gain_signal or credible)))
        allowance=regular+(self.limit-regular)//2 if near else regular
        # 每题前十六次仍可动用共享名额，给探索较少的题目保留机会。
        task_share=max(1,self.limit//(2*self.task_count))
        if self.used[problem]>=task_share and spent>=allowance:
            return "SKIP_RESERVE"
        if action=="SKIP_MODEL" and not near:
            return action
        self.used[problem]+=1
        return "KEEP"
