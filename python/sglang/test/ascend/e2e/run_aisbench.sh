#!/bin/bash

set -e

show_usage() {
    echo -e "\033[31mUsage:\033[0m"
    echo "  $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mode           Benchmark mode: perf | accuracy (required)"
    echo "  --ip             Server IP address (required)"
    echo "  --port           Server port (required)"
    echo "  --model          Model name (required)"
    echo "  --model-path     Model path (required)"
    echo "  --dataset-type   Dataset type: gsm8k | sharegpt | mm-custom-gen (required)"
    echo "  --dataset-name   Dataset name (default: auto-generated if mode=perf; required if mode=accuracy)"
    echo "  --dataset-path   Dataset path (automatic if not provided)"
    echo "  --input-len      Input token length (required if mode=perf)"
    echo "  --output-len     Output token length (required)"
    echo "  --batch-size     Batch size (required)"
    echo "  --num-prompts    Number of prompts (default: 128)"
    echo "  --output-path    Output path (default: ./result)"
    echo ""
    echo "Example:"
    echo "  $0 --mode perf --ip 127.0.0.1 --port 54321 --model Qwen2-7B-Instruct \\"
    echo "     --model-path /models/qwen --dataset-type gsm8k"
    exit 1
}

MODE=""
IP=""
PORT=""
MODEL=""
MODEL_PATH=""
DATASET_TYPE=""
DATASET_NAME=""
DATASET_PATH=""
INPUT_LEN="2048"
OUTPUT_LEN="8192"
BATCH_SIZE=""
NUM_PROMPTS="128"
OUTPUT_PATH="./result"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --ip)
            IP="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset-type)
            DATASET_TYPE="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --input-len)
            INPUT_LEN="$2"
            shift 2
            ;;
        --output-len)
            OUTPUT_LEN="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --output-path)
            OUTPUT_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            ;;
    esac
done

if [ -z "$MODE" ] || [ -z "$IP" ] || [ -z "$PORT" ] || [ -z "$MODEL" ] || [ -z "$MODEL_PATH" ] || [ -z "$DATASET_TYPE" ] || [ -z "$BATCH_SIZE" ]; then
    echo "Error: Missing required parameters."
    show_usage
fi

install_aisbench() {
    echo "===== Install aisbench in virtual env - Begin ====="
    PYTHON_ENV_FOR_AISBENCH=test_env_aisbench
    PIP_FOR_AISBENCH=${PYTHON_ENV_FOR_AISBENCH}/bin/pip
    python -m venv ${PYTHON_ENV_FOR_AISBENCH}
    AISBENCH_SOURCE_PATH=/root/.cache/.cache/benchmark
    AISBENCH_PKG_PATH=/root/.cache/.cache/aisbench-packages
    if [ ! -d "${AISBENCH_SOURCE_PATH}" ]; then
        echo "The aisbench source does not exist: ${AISBENCH_SOURCE_PATH}."
        echo "git clone https://github.com/AISBench/benchmark.git"
        git clone https://github.com/AISBench/benchmark.git
        AISBENCH_SOURCE_PATH="./benchmark/"
    fi
    pip_mirror_source="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    if [ ! -d "${AISBENCH_PKG_PATH}" ]; then
        echo "The dependent aisbench package does not exist: ${AISBENCH_PKG_PATH}."
        echo "Install aisbench online."
        ${PIP_FOR_AISBENCH} install -U pip -i ${pip_mirror_source}
        ${PIP_FOR_AISBENCH} install -e ${AISBENCH_SOURCE_PATH} --use-pep517 -i ${pip_mirror_source}
        ${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/api.txt -i ${pip_mirror_source}
        ${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/extra.txt -i ${pip_mirror_source}
        if [ "${DATASET_TYPE}" == "bfcl" ];then
            ${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/datasets/bfcl_dependencies.txt -i ${pip_mirror_source}
        fi
    else
        echo "Install aisbench locally."
        ${PIP_FOR_AISBENCH} install -U pip --no-index --find-links=${AISBENCH_PKG_PATH}
        ${PIP_FOR_AISBENCH} install -e ${AISBENCH_SOURCE_PATH} --use-pep517 --no-index --find-links=${AISBENCH_PKG_PATH}
        ${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/api.txt --no-index --find-links=${AISBENCH_PKG_PATH}
        ${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/extra.txt --no-index --find-links=${AISBENCH_PKG_PATH}
        if [ "${DATASET_TYPE}" == "bfcl" ];then
            ${PIP_FOR_AISBENCH} install -r ${AISBENCH_SOURCE_PATH}/requirements/datasets/bfcl_dependencies.txt --no-index --find-links=${AISBENCH_PKG_PATH}
        fi
    fi
    echo "===== Install aisbench in virtual env - End ====="
}

install_aisbench

CMD="ais_bench "

AISBENCH_CUSTOM_CONFIG_PATH=/tmp/ais_configs

MODEL_CONFIG_PATH=${AISBENCH_CUSTOM_CONFIG_PATH}/models
mkdir -p ${MODEL_CONFIG_PATH}
TMP_CFG=vllm_api_${MODEL}
DATASETS_CONFIG_PATH=${AISBENCH_CUSTOM_CONFIG_PATH}/datasets
mkdir -p ${DATASETS_CONFIG_PATH}

GSM8K_TRAIN_FILE="/root/.cache/modelscope/hub/datasets/grade_school_math/train.jsonl"
if [ ! -f "${GSM8K_TRAIN_FILE}" ];then
  ${PIP_FOR_AISBENCH} install modelscope -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  ${PYTHON_ENV_FOR_AISBENCH}/bin/python -c "
from modelscope import MsDataset
ds = MsDataset.load('AI-ModelScope/gsm8k', split='train')
ds.to_json('/root/.cache/modelscope/hub/datasets/grade_school_math/train.jsonl')
"
fi

function gen_model_config_file_vllm_api_stream_chat() {
  model_config_file=${MODEL_CONFIG_PATH}/${TMP_CFG}.py
  echo "Writing model config info into file: ${model_config_file}"
  if [ "$MODE" == "perf" ]; then
    generation_kwargs="dict(temperature=0,ignore_eos=True)"
  elif [ "$MODE" == "accuracy" ]; then
    generation_kwargs="dict(temperature=0,seed=1234)"
  else
    echo "Error: Unknown mode: $MODE."
    show_usage
  fi

  cat > "${model_config_file}" << EOF
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="$MODEL_PATH",
        model="$MODEL",
        stream=True,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="$IP",
        host_port=$PORT,
        url="",
        max_out_len=${OUTPUT_LEN},
        batch_size=$BATCH_SIZE,
        trust_remote_code=True,
        generation_kwargs=${generation_kwargs},
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
EOF
  echo "============== ${model_config_file} - Begin =============="
  echo "$(cat ${model_config_file})"
  echo "============== ${model_config_file} - End ================"
}

function gen_model_config_file_vllm_api_function_call_chat() {
  model_config_file=${MODEL_CONFIG_PATH}/${TMP_CFG}.py
  echo "Writing model config info into file: ${model_config_file}"
  if [ "$MODE" == "perf" ]; then
    generation_kwargs="dict(temperature=0,ignore_eos=True)"
  elif [ "$MODE" == "accuracy" ]; then
    generation_kwargs="dict(temperature=0.01,seed=1234)"
  else
    echo "Error: Unknown mode: $MODE."
    show_usage
  fi

  cat > "${model_config_file}" << EOF
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-function-call-chat",
        path="$MODEL_PATH",
        model="$MODEL",
        request_rate=0,
        retry=2,
        api_key="",
        host_ip="$IP",
        host_port=$PORT,
        url="",
        max_out_len=$OUTPUT_LEN,
        returns_tool_calls=True,
        batch_size=$BATCH_SIZE,
        trust_remote_code=False,
        generation_kwargs=${generation_kwargs},
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
EOF
  echo "============== ${model_config_file} - Begin =============="
  echo "$(cat ${model_config_file})"
  echo "============== ${model_config_file} - End ================"
}

function gen_dataset_mm_custom_config_file() {
  dataset_file=${DATASETS_CONFIG_PATH}/${DATASET_NAME}.py
  echo "Writing mm_custom config info into file: ${dataset_file}"
  cat > "${dataset_file}" << EOF
from ais_bench.benchmark.openicl.icl_prompt_template.icl_prompt_template_mm import MMPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MMCustomDataset, MMCustomEvaluator


mm_custom_reader_cfg = dict(
    input_columns=['question', 'image', 'video', 'audio'],
    output_column='answer'
)


mm_custom_infer_cfg = dict(
    prompt_template=dict(
        type=MMPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt_mm={
                    "text": {"type": "text", "text": "{question}"},
                    "image": {"type": "image_url", "image_url": {"url": "file://{image}"}},
                    "video": {"type": "video_url", "video_url": {"url": "file://{video}"}},
                    "audio": {"type": "audio_url", "audio_url": {"url": "file://{audio}"}},
                })
            ]
            )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mm_custom_eval_cfg = dict(
    evaluator=dict(type=MMCustomEvaluator)
)

mm_custom_datasets = [
    dict(
        abbr='mm_custom',
        type=MMCustomDataset,
        path="$DATASET_PATH",
        mm_type="path",
        num_frames=5,
        reader_cfg=mm_custom_reader_cfg,
        infer_cfg=mm_custom_infer_cfg,
        eval_cfg=mm_custom_eval_cfg,
        k=1,
        n=1,
    )
]
EOF
  echo "============== ${dataset_file} - Begin =============="
  echo "$(cat ${dataset_file})"
  echo "============== ${dataset_file} - End ================"
}

function gen_dataset_gsm8k_config_file() {
  dataset_dir=$1
  dataset_config_file=${DATASETS_CONFIG_PATH}/${DATASET_NAME}.py
  echo "Writing gsm8k config info into file: ${dataset_config_file}"
  cat > "${dataset_config_file}" << EOF
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess, Gsm8kEvaluator
gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template="{question}"),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

gsm8k_eval_cfg = dict(evaluator=dict(type=Gsm8kEvaluator),
                      pred_role='BOT',
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path="${dataset_dir}",
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]
EOF

  echo "============== ${dataset_config_file} - Begin =============="
  echo "$(cat ${dataset_config_file})"
  echo "============== ${dataset_config_file} - End ================"
}

function gen_dataset_sharegpt_config_file() {
  dataset_file=$1
  dataset_config_file=${DATASETS_CONFIG_PATH}/${DATASET_NAME}.py
  echo "Writing sharegpt config info into file: ${dataset_config_file}"
  cat > "${dataset_config_file}" << EOF
from ais_bench.benchmark.openicl.icl_prompt_template import MultiTurnPromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import MultiTurnGenInferencer
from ais_bench.benchmark.datasets import ShareGPTDataset, ShareGPTEvaluator, math_postprocess_v2


sharegpt_reader_cfg = dict(
    input_columns=["question", "answer"],
    output_column="answer"
)


sharegpt_infer_cfg = dict(
    prompt_template=dict(
        type=MultiTurnPromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="{question}"),
                dict(role="BOT", prompt="{answer}"),
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=MultiTurnGenInferencer, infer_mode="every") # Default using "every" mode, Supports: "last", "every", "every_with_gt"
)

sharegpt_eval_cfg = dict(
    evaluator=dict(type=ShareGPTEvaluator)
)

sharegpt_datasets = [
    dict(
        abbr='sharegpt',
        type=ShareGPTDataset,
        disable_shuffle=True,
        path="${dataset_file}",
        reader_cfg=sharegpt_reader_cfg,
        infer_cfg=sharegpt_infer_cfg,
        eval_cfg=sharegpt_eval_cfg
    )
]
EOF
  echo "============== ${dataset_config_file} - Begin =============="
  echo "$(cat ${dataset_config_file})"
  echo "============== ${dataset_config_file} - End ================"
}

if [ "$MODE" == "perf" ];then
    if [ "$DATASET_TYPE" == "sharegpt" ]; then
        dataset_file=$DATASET_PATH
        if [ ! -f "${dataset_file}" ]; then
            echo "The sharegpt dataset file does not exist: ${DATASET_PATH}."
            exit 1
        fi
        DATASET_NAME=sharegpt_custom_${MODEL}
        gen_dataset_sharegpt_config_file "${dataset_file}"
        echo "Use dataset: ${DATASET_NAME}"
        gen_model_config_file_vllm_api_stream_chat
        CMD="${CMD} --config-dir ${AISBENCH_CUSTOM_CONFIG_PATH} --models $TMP_CFG --datasets ${DATASET_NAME} --debug --summarizer default_perf --mode perf --num-prompts $NUM_PROMPTS --work-dir $OUTPUT_PATH "
    elif [ "$DATASET_TYPE" == "mm-custom-gen" ]; then
        if [ ! -f "$DATASET_PATH" ]; then
            echo "The mm-custom-gen dataset file does not exist: ${DATASET_PATH}."
            exit 1
        fi
        DATASET_NAME=mm_custom_gen_${MODEL}
        gen_dataset_mm_custom_config_file
        echo "Use dataset: ${DATASET_NAME}"
        gen_model_config_file_vllm_api_stream_chat
        CMD="${CMD} --config-dir ${AISBENCH_CUSTOM_CONFIG_PATH} --models $TMP_CFG --datasets ${DATASET_NAME} --mode perf --num-prompts $NUM_PROMPTS --work-dir $OUTPUT_PATH "
    elif [ "$DATASET_TYPE" == "custom-gen" ]; then
        dataset_file=$DATASET_PATH
        DATASET_NAME=gsm8k_gen_${MODEL}
        gen_dataset_custom_config_file "${dataset_file}"
        echo "Use dataset: ${DATASET_NAME}"
        gen_model_config_file_vllm_api_stream_chat
        CMD="${CMD} --config-dir ${AISBENCH_CUSTOM_CONFIG_PATH} --models $TMP_CFG --datasets ${DATASET_NAME} --summarizer default_perf --mode perf --num-prompts $NUM_PROMPTS --work-dir $OUTPUT_PATH "
    elif [ "$DATASET_TYPE" == "gsm8k" ]; then
        dataset_file=$DATASET_PATH
        if [ ! -f "${dataset_file}" ]; then
            echo "The gsm8k dataset file does not exist: ${DATASET_PATH}."
            exit 1
        fi
        dataset_dir=$(dirname "$DATASET_PATH")
        if [ -f "${GSM8K_TRAIN_FILE}" ]; then
            cp ${GSM8K_TRAIN_FILE} ${dataset_dir}
        else
            echo "Warning: GSM8K train file not found at ${GSM8K_TRAIN_FILE}"
        fi
        DATASET_NAME=gsm8k_custom_${MODEL}
        gen_dataset_gsm8k_config_file "${dataset_dir}"
        echo "Use dataset: ${DATASET_NAME}, dataset_file: ${dataset_file}"
        gen_model_config_file_vllm_api_stream_chat
        CMD="${CMD} --config-dir ${AISBENCH_CUSTOM_CONFIG_PATH} --models $TMP_CFG --datasets ${DATASET_NAME} --debug --summarizer default_perf --mode perf --num-prompts $NUM_PROMPTS --work-dir $OUTPUT_PATH "
    else
        echo "The dataset type $DATASET_TYPE is not supported."
        exit 1
    fi
    echo "IP: $IP | Port: $PORT | Model: $MODEL | Model Path: $MODEL_PATH"
    echo "Input tokens: ${INPUT_LEN} | Output tokens: ${OUTPUT_LEN} | Batch size: ${BATCH_SIZE} | Prompts num: ${NUM_PROMPTS}"

elif [ "$MODE" == "accuracy" ]; then
    if [ -z "$DATASET_NAME" ]; then
        echo "The dataset name is not provided: ${DATASET_NAME}."
        exit 1
    fi
    if [ "$DATASET_TYPE" == "bfcl" ]; then
        gen_model_config_file_vllm_api_function_call_chat
    else
        gen_model_config_file_vllm_api_stream_chat
    fi
    CMD="${CMD} --config-dir ${AISBENCH_CUSTOM_CONFIG_PATH} --models $TMP_CFG --datasets ${DATASET_NAME} --work-dir $OUTPUT_PATH "
    echo "IP: $IP | Port: $PORT | Model: $MODEL | Model Path: $MODEL_PATH"
    echo "max_out_len: ${OUTPUT_LEN} | batch_size: ${BATCH_SIZE} | datasets: ${DATASET_NAME}"

else
    echo "The mode $MODE is not supported."
    exit 1
fi

source ${PYTHON_ENV_FOR_AISBENCH}/bin/activate
echo "Run command: ${CMD}"
eval "${CMD}"
