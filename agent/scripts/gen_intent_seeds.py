"""生成意图语料草稿：模板变体 + 人工撰写的边界用例。

产出 data/intent/seeds/seed_questions.jsonl，每行：
    {id, text, context, hint_labels, source, draft}

hint_labels 只是"按构造方式推断的提示标签"，不是最终标注；
最终标签由 label_intent.py 用大模型预标注 + 人工校准产出。

用法（在 agent/ 目录下执行）：
    python scripts/gen_intent_seeds.py
    python scripts/gen_intent_seeds.py --show 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT_ROOT))

from agent.db import DATA_DIR  # noqa: E402

SEEDS_DIR = DATA_DIR / "intent" / "seeds"
SEEDS_PATH = SEEDS_DIR / "seed_questions.jsonl"

PRODUCTS: tuple[str, ...] = (
    "草莓", "车厘子", "蓝莓", "树莓", "牛油果", "黄桃", "芒果", "香蕉", "火龙果",
    "榴莲", "西瓜", "葡萄", "橙子", "猕猴桃", "土豆", "洋葱", "胡萝卜", "白萝卜",
    "叶菜", "西兰花", "玉米", "鸡蛋", "冷鲜肉", "冷冻肉", "活虾", "三文鱼", "生蚝",
    "带鱼", "蛤蜊", "贝贝南瓜",
)

# 每个意图的模板；带 {product} 的会按商品展开
TEMPLATES: dict[str, tuple[str, ...]] = {
    "query_order": (
        "{product}的订单到哪了",
        "帮我查一下{product}的物流",
        "我买的{product}发货了吗",
        "我的订单到哪了",
        "订单显示签收了吗",
    ),
    "urge_logistics": (
        "{product}怎么还没发货",
        "{product}太慢了能不能催一下",
        "帮我催一下{product}的快递",
        "都三天了怎么还不发货",
        "能不能催一下派送",
    ),
    "bad_fruit_claim": (
        "{product}到货坏了一半",
        "{product}发霉了怎么赔",
        "{product}有异味能赔吗",
        "买的{product}烂了",
        "{product}里面有个坏的怎么处理",
    ),
    "short_weight": (
        "{product}重量不够",
        "{product}少称了怎么办",
        "{product}数量少了一盒",
        "说好一斤只有八两",
        "缺斤少两怎么赔",
    ),
    "refund_consult": (
        "退款多久到账",
        "退款失败了怎么办",
        "{product}退款原路返回吗",
        "退款进度怎么查",
        "退款要几个工作日",
        "退款到账了吗",
        "退款被驳回了怎么办",
        "退款的流程是什么",
        "退款能加急吗",
    ),
    "invoice": (
        "怎么开发票",
        "{product}的订单能开电子发票吗",
        "发票抬头写错了怎么办",
        "开票要多久",
        "能开专用发票吗",
        "发票怎么申请",
        "能补开发票吗",
        "发票弄丢了怎么办",
        "可以改发票金额吗",
    ),
    "quality_qa": (
        "{product}怎么保存",
        "{product}能放多久",
        "{product}需要冷藏吗",
        "{product}冷冻后还能吃吗",
        "生鲜能无理由退货吗",
    ),
    "complaint": (
        "我要投诉你们",
        "客服态度太差我要投诉",
        "再这样就给差评",
        "我要曝光你们",
        "{product}的商家我要投诉",
        "我要投诉客服",
        "你们的服务太差了",
        "再不处理我就投诉到有关部门",
        "这个商家必须处罚",
        "我要在小红书上曝光",
    ),
    "chitchat": (
        "今天天气怎么样",
        "讲个笑话",
        "你叫什么名字",
        "你喜欢吃什么",
        "帮我写一首诗",
        "你多大了",
        "今天星期几",
        "推荐一部电影",
        "你会做什么",
    ),
    "human_handoff": (
        "我要转人工",
        "找人工客服",
        "别机器人了叫真人",
        "帮我接人工客服",
        "转人工客服",
        "人工客服在吗",
        "我要和真人说话",
        "机器人听不懂，转人工",
        "麻烦转接客服人员",
        "给我转真人客服",
    ),
}

# 人工撰写的边界用例：多轮指代、易混类别、多意图、域外
HANDCRAFTED: tuple[dict, ...] = (
    {"text": "那它呢", "context": {"last_question": "草莓怎么保存", "product": "草莓"}, "hint_labels": ["quality_qa"]},
    {"text": "这个能退吗", "context": {"last_question": "车厘子怎么保存", "product": "车厘子"}, "hint_labels": ["quality_qa"]},
    {"text": "它能放多久", "context": {"last_question": "冷鲜肉怎么保存", "product": "冷冻肉"}, "hint_labels": ["quality_qa"]},
    {"text": "这个怎么赔", "context": {"last_question": "车厘子到货坏了", "product": "车厘子"}, "hint_labels": ["bad_fruit_claim"]},
    {"text": "那为什么少了这么多", "context": {"last_question": "重量不够", "product": "牛油果"}, "hint_labels": ["short_weight"]},
    {"text": "三天了怎么还不发货", "hint_labels": ["urge_logistics"]},
    {"text": "我的订单到哪了", "hint_labels": ["query_order"]},
    {"text": "退款多久到账", "hint_labels": ["refund_consult"]},
    {"text": "生鲜可以无理由退货吗", "hint_labels": ["quality_qa"]},
    {"text": "草莓坏了一半怎么赔", "hint_labels": ["bad_fruit_claim"]},
    {"text": "草莓少称了半斤", "hint_labels": ["short_weight"]},
    {"text": "我要投诉你们", "hint_labels": ["complaint"]},
    {"text": "我要转人工", "hint_labels": ["human_handoff"]},
    {"text": "投诉你们客服，顺便转人工", "hint_labels": ["human_handoff", "complaint"]},
    {"text": "车厘子坏了而且还没发货", "hint_labels": ["bad_fruit_claim", "urge_logistics"]},
    {"text": "东西烂了而且少秤了", "hint_labels": ["bad_fruit_claim", "short_weight"]},
    {"text": "退款的发票怎么开", "hint_labels": ["invoice"]},
    {"text": "坏果的赔付标准在哪看", "hint_labels": ["quality_qa"]},
    {"text": "称重照片需要拍成什么样", "hint_labels": ["quality_qa"]},
    {"text": "照片传不上来可以文字描述吗", "hint_labels": ["quality_qa"]},
    {"text": "赔付超过两百为什么要人工审核", "hint_labels": ["quality_qa"]},
    {"text": "你们几点下班", "hint_labels": ["chitchat"]},
    {"text": "帮我写一首诗", "hint_labels": ["chitchat"]},
    {"text": "今天股票涨了吗", "hint_labels": ["chitchat"]},
    {"text": "你是谁", "hint_labels": ["chitchat"]},
    {"text": "帮我订一张机票", "hint_labels": ["chitchat"]},
    {"text": "快递显示签收但我没收到", "hint_labels": ["query_order", "complaint"]},
    {"text": "为什么只能赔一半", "hint_labels": ["quality_qa"]},
    {"text": "不想要优惠券能直接退钱吗", "hint_labels": ["refund_consult"]},
    {"text": "冷冻肉放三个月还能吃吗", "hint_labels": ["quality_qa"]},
    {"text": "香蕉放冰箱可以吗", "hint_labels": ["quality_qa"]},
    {"text": "土豆发芽了还能吃吗", "hint_labels": ["quality_qa"]},
    {"text": "物流一天没更新怎么办", "hint_labels": ["urge_logistics"]},
    {"text": "配送超时怎么补偿", "hint_labels": ["quality_qa"]},
    {"text": "可以修改收货地址吗", "hint_labels": ["query_order"]},
    {"text": "发票能重新开吗", "hint_labels": ["invoice"]},
    {"text": "异物理赔要多久", "hint_labels": ["bad_fruit_claim"]},
    {"text": "你们家水果好吃吗", "hint_labels": ["chitchat"]},
    {"text": "帮我接真人", "hint_labels": ["human_handoff"]},
    {"text": "少了一两", "hint_labels": ["short_weight"]},
)


def build_rows() -> list[dict]:
    rows: list[dict] = []
    counter = 0

    def add(text: str, hint_labels: list[str], context=None, source: str = "template") -> None:
        nonlocal counter
        counter += 1
        rows.append(
            {
                "id": f"seed-{counter:04d}",
                "text": text,
                "context": context,
                "hint_labels": hint_labels,
                "source": source,
                "draft": True,
            }
        )

    for label, templates in TEMPLATES.items():
        for template in templates:
            if "{product}" in template:
                for product in PRODUCTS:
                    add(template.format(product=product), [label])
            else:
                add(template, [label])

    for item in HANDCRAFTED:
        add(
            item["text"],
            list(item["hint_labels"]),
            context=item.get("context"),
            source="handcrafted",
        )
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    kept: list[dict] = []
    for row in rows:
        key = (row["text"].strip(), json.dumps(row.get("context"), ensure_ascii=False, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        kept.append({**row, "text": row["text"].strip()})
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="生成意图语料草稿")
    parser.add_argument("--output", type=Path, default=SEEDS_PATH)
    parser.add_argument("--show", type=int, default=0, help="抽样打印多少条")
    args = parser.parse_args()

    rows = dedupe(build_rows())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )

    counts = Counter(row["hint_labels"][0] for row in rows)
    print(f"语料草稿已写入：{args.output}")
    print(f"总条数：{len(rows)}（模板变体 {sum(1 for r in rows if r['source'] == 'template')}，"
          f"人工撰写 {sum(1 for r in rows if r['source'] == 'handcrafted')}）")
    print("\n按首要意图分布：")
    for label, count in counts.most_common():
        print(f"  {label:<18}{count}")

    multi = [row for row in rows if len(row["hint_labels"]) > 1]
    print(f"\n多意图样本：{len(multi)} 条；带上下文样本："
          f"{sum(1 for row in rows if row.get('context'))} 条")

    if args.show:
        print(f"\n抽样 {args.show} 条：")
        for row in rows[: args.show]:
            print(f"  [{row['hint_labels'][0]}] {row['text']}")


if __name__ == "__main__":
    main()
