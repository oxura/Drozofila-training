"""One neural checkpoint for sums, logic and variable programs; no evaluator."""
import argparse
import json
from pathlib import Path
import torch
from stage6_model import load_model
from stage6_engine import generate
from stage6_tokens import encode

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description='Общая нейронная модель для формальных задач: сумма, логика, код.')
    p.add_argument('prompt');p.add_argument('--checkpoint',type=Path);p.add_argument('--json',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=512);args=p.parse_args();torch.set_num_threads(2)
    try:
        if len(encode(args.prompt))>2048:raise ValueError('Вход ограничен 2048 символами.')
        if not 1<=args.max_new_tokens<=2048:raise ValueError('Лимит генерации должен быть 1–2048.')
        checkpoint=args.checkpoint
        if checkpoint is None:
            selection=json.loads((ROOT/'results/stage6/default_model.json').read_text())
            checkpoint=ROOT/selection['checkpoint']
        model,saved=load_model(checkpoint)
        out=generate(model,[args.prompt],max_new_tokens=args.max_new_tokens)[0]
        mode=saved['config']['mode'];answer=out['text'] if mode=='direct' else out['text'].rsplit('|',1)[-1]
        result=dict(**out,answer=answer,mode=mode,condition=model.condition,copy=model.use_copy,
                    checkpoint=str(checkpoint),scope='Формальные задачи; вывод модели без подстановки ответа из вычислителя.')
        if args.json:print(json.dumps(result,ensure_ascii=False,indent=2))
        else:
            if mode!='direct' and '|' in out['text']:print('Шаги модели: '+out['text'].rsplit('|',1)[0])
            print('Ответ модели: '+answer)
            if not out['halted']:print('Генерация достигла заданного лимита.')
    except (ValueError,FileNotFoundError) as error:p.error(str(error))


if __name__=='__main__':main()
