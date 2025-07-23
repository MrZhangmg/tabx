import json
import math

def compute_ifd(loss):
    loss_ins, loss_out, loss_ins_cond, loss_out_cond = loss
    ifd = math.exp(loss_out_cond - loss_out)      # IFD = exp(L(y|x) - L(y))
    r_ifd = math.exp(loss_ins_cond - loss_ins)    # r-IFD = exp(L(x|y) - L(x))
    return ifd, r_ifd

def load_jsonl(path):
    data_dict = {}
    with open(path, 'r', encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            sample_id = item.get('id')
            if not sample_id or 'loss' not in item or len(item['loss']) < 4:
                continue
            data_dict[sample_id] = item
    return data_dict

def load_raw_json(path):
    with open(path, 'r', encoding="utf-8") as f:
        data = json.load(f)
    return {d['id']: d for d in data if 'id' in d}

def load_jsonl_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

def ifd_based_decision_and_merge(
    metrics_file1_path,
    metrics_file2_path,
    raw_file1_path,
    raw_file2_path,
    retained_path,
    decision_output_path,
    final_merged_output_path
):
    # 加载 metrics 文件（包含 loss）
    metrics1 = load_jsonl(metrics_file1_path)
    metrics2 = load_jsonl(metrics_file2_path)
    common_ids = set(metrics1.keys()) & set(metrics2.keys())

    # 加载原始数据文件
    raw1 = load_raw_json(raw_file1_path)
    raw2 = load_raw_json(raw_file2_path)

    # 加载保留的数据
    retained_data = load_jsonl_list(retained_path)

    decision_result = []
    merged_result = []

    for sample_id in common_ids:
        m1 = metrics1[sample_id]
        m2 = metrics2[sample_id]
        ifd1, r_ifd1 = compute_ifd(m1['loss'])
        ifd2, r_ifd2 = compute_ifd(m2['loss'])

        # 判断策略
        metric_mode = raw1.get(sample_id, raw2.get(sample_id, {})).get("metric", "IFD")
        if metric_mode == "r-IFD":
            chosen_file = 'file1' if r_ifd1 <= r_ifd2 else 'file2'
        else:
            chosen_file = 'file1' if ifd1 >= ifd2 else 'file2'

        selected_data = raw1.get(sample_id) if chosen_file == 'file1' else raw2.get(sample_id)
        if selected_data:
            cleaned_data = {k: v for k, v in selected_data.items() if k != "metric"}
            decision_result.append({'id': sample_id, 'file': chosen_file})
            merged_result.append(cleaned_data)

    # 合并保留数据
    final_merged = merged_result + retained_data

    # 保存结果
    write_jsonl(decision_result, decision_output_path)

    with open(final_merged_output_path, 'w', encoding='utf-8') as f:
        json.dump(final_merged, f, indent=2, ensure_ascii=False)
