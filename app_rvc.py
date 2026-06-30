import gradio as gr
from soni_translate.logging_setup import (
    logger,
    set_logging_level,
    configure_logging_libs,
); configure_logging_libs() # noqa
import whisperx
import torch
import os
from soni_translate.audio_segments import create_translated_audio
from soni_translate.text_to_speech import (
    audio_segmentation_to_voice,
    edge_tts_voices_list,
    coqui_xtts_voices_list,
    piper_tts_voices_list,
    create_wav_file_vc,
    accelerate_segments,
)
from soni_translate.translate_segments import (
    translate_text,
    TRANSLATION_PROCESS_OPTIONS,
    DOCS_TRANSLATION_PROCESS_OPTIONS
)
from soni_translate.preprocessor import (
    audio_video_preprocessor,
    audio_preprocessor,
)
from soni_translate.postprocessor import (
    OUTPUT_TYPE_OPTIONS,
    DOCS_OUTPUT_TYPE_OPTIONS,
    sound_separate,
    get_no_ext_filename,
    media_out,
    get_subtitle_speaker,
)
from soni_translate.language_configuration import (
    LANGUAGES,
    UNIDIRECTIONAL_L_LIST,
    LANGUAGES_LIST,
    BARK_VOICES_LIST,
    VITS_VOICES_LIST,
    OPENAI_TTS_MODELS,
)
from soni_translate.utils import (
    remove_files,
    download_list,
    upload_model_list,
    download_manager,
    run_command,
    is_audio_file,
    is_subtitle_file,
    copy_files,
    get_valid_files,
    get_link_list,
    remove_directory_contents,
)
from soni_translate.mdx_net import (
    UVR_MODELS,
    MDX_DOWNLOAD_LINK,
    mdxnet_models_dir,
)
from soni_translate.speech_segmentation import (
    ASR_MODEL_OPTIONS,
    COMPUTE_TYPE_GPU,
    COMPUTE_TYPE_CPU,
    find_whisper_models,
    transcribe_speech,
    align_speech,
    diarize_speech,
    diarization_models,
)
from soni_translate.text_multiformat_processor import (
    BORDER_COLORS,
    srt_file_to_segments,
    document_preprocessor,
    determine_chunk_size,
    plain_text_to_segments,
    segments_to_plain_text,
    process_subtitles,
    linguistic_level_segments,
    break_aling_segments,
    doc_to_txtximg_pages,
    page_data_to_segments,
    update_page_data,
    fix_timestamps_docs,
    create_video_from_images,
    merge_video_and_audio,
)
from soni_translate.languages_gui import language_data, news
import os
import copy
import logging
import json
from pydub import AudioSegment
from voice_main import ClassVoices
import argparse
import time
import hashlib
import sys

directories = [
    "downloads",
    "logs",
    "weights",
    "clean_song_output",
    "_XTTS_",
    f"audio2{os.sep}audio",
    "audio",
    "outputs",
    "temp",
]
[
    os.makedirs(directory)
    for directory in directories
    if not os.path.exists(directory)
]


class TTS_Info:
    def __init__(self, piper_enabled, xtts_enabled):
        self.list_edge = edge_tts_voices_list()
        self.list_bark = list(BARK_VOICES_LIST.keys())
        self.list_vits = list(VITS_VOICES_LIST.keys())
        self.list_openai_tts = OPENAI_TTS_MODELS
        self.piper_enabled = piper_enabled
        self.list_vits_onnx = (
            piper_tts_voices_list() if self.piper_enabled else []
        )
        self.xtts_enabled = xtts_enabled

    def tts_list(self):
        self.list_coqui_xtts = (
            coqui_xtts_voices_list() if self.xtts_enabled else []
        )
        list_tts = self.list_coqui_xtts + sorted(
            self.list_edge
            + self.list_bark
            + self.list_vits
            + self.list_openai_tts
            + self.list_vits_onnx
        )
        return list_tts


def prog_disp(msg, percent, is_gui, progress=None):
    logger.info(msg)
    if is_gui:
        progress(percent, desc=msg)


def warn_disp(wrn_lang, is_gui):
    logger.warning(wrn_lang)
    if is_gui:
        gr.Warning(wrn_lang)


class SoniTrCache:
    def __init__(self):
        self.cache = {
            'media': [[]],
            'refine_vocals': [],
            'transcript_align': [],
            'break_align': [],
            'diarize': [],
            'translate': [],
            'subs_and_edit': [],
            'tts': [],
            'acc_and_vc': [],
            'mix_aud': [],
            'output': []
        }

        self.cache_data = {
            'media': [],
            'refine_vocals': [],
            'transcript_align': [],
            'break_align': [],
            'diarize': [],
            'translate': [],
            'subs_and_edit': [],
            'tts': [],
            'acc_and_vc': [],
            'mix_aud': [],
            'output': []
        }

        self.cache_keys = list(self.cache.keys())
        self.first_task = self.cache_keys[0]
        self.last_task = self.cache_keys[-1]

        self.pre_step = None
        self.pre_params = []

    def set_variable(self, variable_name, value):
        setattr(self, variable_name, value)

    def task_in_cache(self, step: str, params: list, previous_step_data: dict):

        self.pre_step_cache = None

        if step == self.first_task:
            self.pre_step = None

        if self.pre_step:
            self.cache[self.pre_step] = self.pre_params

            # Fill data in cache
            self.cache_data[self.pre_step] = copy.deepcopy(previous_step_data)

        self.pre_params = params
        # logger.debug(f"Step: {str(step)}, Cache params: {str(self.cache)}")
        if params == self.cache[step]:
            logger.debug(f"In cache: {str(step)}")

            # Set the var needed for next step
            # Recovery from cache_data the current step
            for key, value in self.cache_data[step].items():
                self.set_variable(key, copy.deepcopy(value))
                logger.debug(
                    f"Chache load: {str(key)}"
                )

            self.pre_step = step
            return True

        else:
            logger.debug(f"Flush next and caching {str(step)}")
            selected_index = self.cache_keys.index(step)

            for idx, key in enumerate(self.cache.keys()):
                if idx >= selected_index:
                    self.cache[key] = []
                    self.cache_data[key] = {}

            # The last is now previous
            self.pre_step = step
            return False

    def clear_cache(self, media, force=False):

        self.cache["media"] = (
            self.cache["media"] if len(self.cache["media"]) else [[]]
        )

        if media != self.cache["media"][0] or force:

            # Clear cache
            self.cache = {key: [] for key in self.cache}
            self.cache["media"] = [[]]

            logger.info("Cache flushed")


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:18]


def check_openai_api_key():
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "To use GPT for translation, please set up your OpenAI API key "
            "as an environment variable in Linux as follows: "
            "export OPENAI_API_KEY='your-api-key-here'. Or change the "
            "translation process in Advanced settings."
        )


class SoniTranslate(SoniTrCache):
    def __init__(self, cpu_mode=False):
        super().__init__()
        if cpu_mode:
            os.environ["SONITR_DEVICE"] = "cpu"
        else:
            os.environ["SONITR_DEVICE"] = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        self.device = os.environ.get("SONITR_DEVICE")
        self.result_diarize = None
        self.align_language = None
        self.result_source_lang = None
        self.edit_subs_complete = False
        self.voiceless_id = None
        self.burn_subs_id = None

        self.vci = ClassVoices(only_cpu=cpu_mode)

        self.tts_voices = self.get_tts_voice_list()

        logger.info(f"Working in: {self.device}")

    def get_tts_voice_list(self):
        try:
            from piper import PiperVoice  # noqa

            piper_enabled = True
            logger.info("PIPER TTS enabled")
        except Exception as error:
            logger.debug(str(error))
            piper_enabled = False
            logger.info("PIPER TTS disabled")
        try:
            from TTS.api import TTS  # noqa

            xtts_enabled = True
            logger.info("Coqui XTTS enabled")
            logger.info(
                "In this app, by using Coqui TTS (text-to-speech), you "
                "acknowledge and agree to the license.\n"
                "You confirm that you have read, understood, and agreed "
                "to the Terms and Conditions specified at the following "
                "link:\nhttps://coqui.ai/cpml.txt."
            )
            os.environ["COQUI_TOS_AGREED"] = "1"
        except Exception as error:
            logger.debug(str(error))
            xtts_enabled = False
            logger.info("Coqui XTTS disabled")

        self.tts_info = TTS_Info(piper_enabled, xtts_enabled)

        return self.tts_info.tts_list()

    def batch_multilingual_media_conversion(self, *kwargs):
        # logger.debug(str(kwargs))

        media_file_arg = kwargs[0] if kwargs[0] is not None else []

        link_media_arg = kwargs[1]
        link_media_arg = [x.strip() for x in link_media_arg.split(',')]
        link_media_arg = get_link_list(link_media_arg)

        path_arg = kwargs[2]
        path_arg = [x.strip() for x in path_arg.split(',')]
        path_arg = get_valid_files(path_arg)

        edit_text_arg = kwargs[32]
        get_text_arg = kwargs[33]

        is_gui_arg = kwargs[-1]

        kwargs = kwargs[3:]

        media_batch = media_file_arg + link_media_arg + path_arg
        media_batch = list(filter(lambda x: x != "", media_batch))
        media_batch = media_batch if media_batch else [None]
        logger.debug(str(media_batch))

        remove_directory_contents("outputs")

        if edit_text_arg or get_text_arg:
            return self.multilingual_media_conversion(
                media_batch[0], "", "", *kwargs
            )

        if "SET_LIMIT" == os.getenv("DEMO"):
            media_batch = [media_batch[0]]

        result = []
        for media in media_batch:
            # Call the nested function with the parameters
            output_file = self.multilingual_media_conversion(
                media, "", "", *kwargs
            )

            if isinstance(output_file, str):
                output_file = [output_file]
            result.extend(output_file)

            if is_gui_arg and len(media_batch) > 1:
                gr.Info(f"Done: {os.path.basename(output_file[0])}")

        return result

    def multilingual_media_conversion(
        self,
        media_file=None,
        link_media="",
        directory_input="",
        YOUR_HF_TOKEN="",
        preview=False,
        transcriber_model="large-v3",
        batch_size=4,
        compute_type="auto",
        origin_language="Automatic detection",
        target_language="English (en)",
        min_speakers=1,
        max_speakers=1,
        tts_voice00="en-US-EmmaMultilingualNeural-Female",
        tts_voice01="en-US-AndrewMultilingualNeural-Male",
        tts_voice02="en-US-AvaMultilingualNeural-Female",
        tts_voice03="en-US-BrianMultilingualNeural-Male",
        tts_voice04="de-DE-SeraphinaMultilingualNeural-Female",
        tts_voice05="de-DE-FlorianMultilingualNeural-Male",
        tts_voice06="fr-FR-VivienneMultilingualNeural-Female",
        tts_voice07="fr-FR-RemyMultilingualNeural-Male",
        tts_voice08="en-US-EmmaMultilingualNeural-Female",
        tts_voice09="en-US-AndrewMultilingualNeural-Male",
        tts_voice10="en-US-EmmaMultilingualNeural-Female",
        tts_voice11="en-US-AndrewMultilingualNeural-Male",
        tts_voice12="en-US-EmmaMultilingualNeural-Female",
        tts_voice13="en-US-AndrewMultilingualNeural-Male",
        tts_voice14="en-US-EmmaMultilingualNeural-Female",
        tts_voice15="en-US-AndrewMultilingualNeural-Male",
        tts_voice16="en-US-EmmaMultilingualNeural-Female",
        tts_voice17="en-US-AndrewMultilingualNeural-Male",
        tts_voice18="en-US-EmmaMultilingualNeural-Female",
        tts_voice19="en-US-AndrewMultilingualNeural-Male",
        video_output_name="",
        mix_method_audio="Adjusting volumes and mixing audio",
        max_accelerate_audio=2.1,
        acceleration_rate_regulation=False,
        volume_original_audio=0.25,
        volume_translated_audio=1.80,
        output_format_subtitle="srt",
        get_translated_text=False,
        get_video_from_text_json=False,
        text_json="{}",
        avoid_overlap=False,
        vocal_refinement=False,
        literalize_numbers=True,
        segment_duration_limit=15,
        diarization_model="pyannote_2.1",
        translate_process="google_translator_batch",
        subtitle_file=None,
        output_type="video (mp4)",
        voiceless_track=False,
        voice_imitation=False,
        voice_imitation_max_segments=3,
        voice_imitation_vocals_dereverb=False,
        voice_imitation_remove_previous=True,
        voice_imitation_method="freevc",
        dereverb_automatic_xtts=True,
        text_segmentation_scale="sentence",
        divide_text_segments_by="",
        soft_subtitles_to_video=True,
        burn_subtitles_to_video=False,
        enable_cache=True,
        custom_voices=False,
        custom_voices_workers=1,
        auto_clone_voices=False,
        enable_lip_sync=False,
        is_gui=False,
        progress=gr.Progress(),
    ):
        if not YOUR_HF_TOKEN:
            YOUR_HF_TOKEN = os.getenv("YOUR_HF_TOKEN")
            if diarization_model == "disable" or max_speakers == 1:
                if YOUR_HF_TOKEN is None:
                    YOUR_HF_TOKEN = ""
            elif not YOUR_HF_TOKEN:
                raise ValueError("No valid Hugging Face token")
            else:
                os.environ["YOUR_HF_TOKEN"] = YOUR_HF_TOKEN

        if (
            "gpt" in translate_process
            or transcriber_model == "OpenAI_API_Whisper"
            or "OpenAI-TTS" in tts_voice00
        ):
            check_openai_api_key()

        if media_file is None:
            media_file = (
                directory_input
                if os.path.exists(directory_input)
                else link_media
            )
        media_file = (
            media_file if isinstance(media_file, str) else media_file.name
        )

        if is_subtitle_file(media_file):
            subtitle_file = media_file
            media_file = ""

        if media_file is None:
            media_file = ""

        if not origin_language:
            origin_language = "Automatic detection"

        if origin_language in UNIDIRECTIONAL_L_LIST and not subtitle_file:
            raise ValueError(
                f"The language '{origin_language}' "
                "is not supported for transcription (ASR)."
            )

        if get_translated_text:
            self.edit_subs_complete = False
        if get_video_from_text_json:
            if not self.edit_subs_complete:
                raise ValueError("Generate the transcription first.")

        if (
            ("sound" in output_type or output_type == "raw media")
            and (get_translated_text or get_video_from_text_json)
        ):
            raise ValueError(
                "Please disable 'edit generate subtitles' "
                f"first to acquire the {output_type}."
            )

        TRANSLATE_AUDIO_TO = LANGUAGES[target_language]
        SOURCE_LANGUAGE = LANGUAGES[origin_language]

        if (
            transcriber_model == "OpenAI_API_Whisper"
            and SOURCE_LANGUAGE == "zh-TW"
        ):
            logger.warning(
                "OpenAI API Whisper only supports Chinese (Simplified)."
            )
            SOURCE_LANGUAGE = "zh"

        if (
            text_segmentation_scale in ["word", "character"]
            and "subtitle" not in output_type
        ):
            wrn_lang = (
                "Text segmentation by words or characters is typically"
                " used for generating subtitles. If subtitles are not the"
                " intended output, consider selecting 'sentence' "
                "segmentation method to ensure optimal results."

            )
            warn_disp(wrn_lang, is_gui)

        if tts_voice00[:2].lower() != TRANSLATE_AUDIO_TO[:2].lower():
            wrn_lang = (
                "Make sure to select a 'TTS Speaker' suitable for"
                " the translation language to avoid errors with the TTS."
            )
            warn_disp(wrn_lang, is_gui)

        if auto_clone_voices:
            tts_voice00 = "_XTTS_/AUTOMATIC.wav"
            tts_voice01 = "_XTTS_/AUTOMATIC.wav"
            tts_voice02 = "_XTTS_/AUTOMATIC.wav"
            tts_voice03 = "_XTTS_/AUTOMATIC.wav"
            tts_voice04 = "_XTTS_/AUTOMATIC.wav"
            tts_voice05 = "_XTTS_/AUTOMATIC.wav"
            tts_voice06 = "_XTTS_/AUTOMATIC.wav"
            tts_voice07 = "_XTTS_/AUTOMATIC.wav"
            tts_voice08 = "_XTTS_/AUTOMATIC.wav"
            tts_voice09 = "_XTTS_/AUTOMATIC.wav"
            tts_voice10 = "_XTTS_/AUTOMATIC.wav"
            tts_voice11 = "_XTTS_/AUTOMATIC.wav"
            tts_voice12 = "_XTTS_/AUTOMATIC.wav"
            tts_voice13 = "_XTTS_/AUTOMATIC.wav"
            tts_voice14 = "_XTTS_/AUTOMATIC.wav"
            tts_voice15 = "_XTTS_/AUTOMATIC.wav"
            tts_voice16 = "_XTTS_/AUTOMATIC.wav"
            tts_voice17 = "_XTTS_/AUTOMATIC.wav"
            tts_voice18 = "_XTTS_/AUTOMATIC.wav"
            tts_voice19 = "_XTTS_/AUTOMATIC.wav"
            voiceless_track = True
            if max_speakers <= 1:
                max_speakers = 20
                min_speakers = 1
                logger.info("Auto-Clone: max_speakers auto-set to 20 for speaker detection")
            logger.info("Auto-Clone Voices enabled: Overriding TTS voices to XTTS AUTOMATIC and enabling voiceless_track.")

        if "_XTTS_" in tts_voice00 and voice_imitation:
            wrn_lang = (
                "When you select XTTS, it is advisable "
                "to disable Voice Imitation."
            )
            warn_disp(wrn_lang, is_gui)

        if custom_voices and voice_imitation:
            wrn_lang = (
                "When you use R.V.C. models, it is advisable"
                " to disable Voice Imitation."
            )
            warn_disp(wrn_lang, is_gui)

        if not media_file and not subtitle_file:
            raise ValueError(
                "Specifify a media or SRT file in advanced settings"
            )

        if subtitle_file:
            subtitle_file = (
                subtitle_file
                if isinstance(subtitle_file, str)
                else subtitle_file.name
            )

        if subtitle_file and SOURCE_LANGUAGE == "Automatic detection":
            raise Exception(
                "To use an SRT file, you need to specify its "
                "original language (Source language)"
            )

        if not media_file and subtitle_file:
            diarization_model = "disable"
            media_file = "audio_support.wav"
            if not get_video_from_text_json:
                remove_files(media_file)
                srt_data = srt_file_to_segments(subtitle_file)
                total_duration = srt_data["segments"][-1]["end"] + 30.
                support_audio = AudioSegment.silent(
                    duration=int(total_duration * 1000)
                )
                support_audio.export(
                    media_file, format="wav"
                )
                logger.info("Supporting audio for the SRT file, created.")

        if "SET_LIMIT" == os.getenv("DEMO"):
            preview = True
            mix_method_audio = "Adjusting volumes and mixing audio"
            transcriber_model = "medium"
            logger.info(
                "DEMO; set preview=True; Generation is limited to "
                "10 seconds to prevent CPU errors. No limitations with GPU.\n"
                "DEMO; set Adjusting volumes and mixing audio\n"
                "DEMO; set whisper model to medium"
            )

        # Check GPU
        if self.device == "cpu" and compute_type not in COMPUTE_TYPE_CPU:
            logger.info("Compute type changed to float32")
            compute_type = "float32"

        base_video_file = "Video.mp4"
        base_audio_wav = "audio.wav"
        dub_audio_file = "audio_dub_solo.ogg"
        vocals_audio_file = "audio_Vocals_DeReverb.wav"
        voiceless_audio_file = "audio_Voiceless.wav"
        mix_audio_file = "audio_mix.mp3"
        vid_subs = "video_subs_file.mp4"
        video_output_file = "video_dub.mp4"

        if os.path.exists(media_file):
            media_base_hash = get_hash(media_file)
        else:
            media_base_hash = media_file
        self.clear_cache(media_base_hash, force=(not enable_cache))

        if not get_video_from_text_json:
            self.result_diarize = (
                self.align_language
            ) = self.result_source_lang = None
            if not self.task_in_cache("media", [media_base_hash, preview], {}):
                if is_audio_file(media_file):
                    prog_disp(
                        "Processing audio...", 0.15, is_gui, progress=progress
                    )
                    audio_preprocessor(preview, media_file, base_audio_wav)
                else:
                    prog_disp(
                        "Processing video...", 0.15, is_gui, progress=progress
                    )
                    audio_video_preprocessor(
                        preview, media_file, base_video_file, base_audio_wav
                    )
                logger.debug("Set file complete.")

            if "sound" in output_type:
                prog_disp(
                    "Separating sounds in the file...",
                    0.50,
                    is_gui,
                    progress=progress
                )
                separate_out = sound_separate(base_audio_wav, output_type)
                final_outputs = []
                for out in separate_out:
                    final_name = media_out(
                        media_file,
                        f"{get_no_ext_filename(out)}",
                        video_output_name,
                        "wav",
                        file_obj=out,
                    )
                    final_outputs.append(final_name)
                logger.info(f"Done: {str(final_outputs)}")
                return final_outputs

            if output_type == "raw media":
                output = media_out(
                    media_file,
                    "raw_media",
                    video_output_name,
                    "wav" if is_audio_file(media_file) else "mp4",
                    file_obj=base_audio_wav if is_audio_file(media_file) else base_video_file,
                )
                logger.info(f"Done: {output}")
                return output

            if not self.task_in_cache("refine_vocals", [vocal_refinement], {}):
                self.vocals = None
                if vocal_refinement:
                    try:
                        from soni_translate.mdx_net import process_uvr_task
                        _, _, _, _, file_vocals = process_uvr_task(
                            orig_song_path=base_audio_wav,
                            main_vocals=False,
                            dereverb=True,
                            remove_files_output_dir=True,
                        )
                        remove_files(vocals_audio_file)
                        copy_files(file_vocals, ".")
                        self.vocals = vocals_audio_file
                    except Exception as error:
                        logger.error(str(error))

            if not self.task_in_cache("transcript_align", [
                subtitle_file,
                SOURCE_LANGUAGE,
                transcriber_model,
                compute_type,
                batch_size,
                literalize_numbers,
                segment_duration_limit,
                (
                    "l_unit"
                    if text_segmentation_scale in ["word", "character"]
                    and subtitle_file
                    else "sentence"
                )
            ], {"vocals": self.vocals}):
                if subtitle_file:
                    prog_disp(
                        "From SRT file...", 0.30, is_gui, progress=progress
                    )
                    audio = whisperx.load_audio(
                        base_audio_wav if not self.vocals else self.vocals
                    )
                    self.result = srt_file_to_segments(subtitle_file)
                    self.result["language"] = SOURCE_LANGUAGE
                else:
                    prog_disp(
                        "Transcribing...", 0.30, is_gui, progress=progress
                    )
                    SOURCE_LANGUAGE = (
                        None
                        if SOURCE_LANGUAGE == "Automatic detection"
                        else SOURCE_LANGUAGE
                    )
                    audio, self.result = transcribe_speech(
                        base_audio_wav if not self.vocals else self.vocals,
                        transcriber_model,
                        compute_type,
                        batch_size,
                        SOURCE_LANGUAGE,
                        literalize_numbers,
                        segment_duration_limit,
                    )
                logger.debug(
                    "Transcript complete, "
                    f"segments count {len(self.result['segments'])}"
                )

                self.align_language = self.result["language"]
                if (
                    not subtitle_file
                    or text_segmentation_scale in ["word", "character"]
                ):
                    prog_disp("Aligning...", 0.45, is_gui, progress=progress)
                    try:
                        if self.align_language in ["vi"]:
                            logger.info(
                                "Deficient alignment for the "
                                f"{self.align_language} language, skipping the"
                                " process. It is suggested to reduce the "
                                "duration of the segments as an alternative."
                            )
                        else:
                            self.result = align_speech(audio, self.result)
                            logger.debug(
                                "Align complete, "
                                f"segments count {len(self.result['segments'])}"
                            )
                    except Exception as error:
                        logger.error(str(error))

            if self.result["segments"] == []:
                raise ValueError("No active speech found in audio")

            if not self.task_in_cache("break_align", [
                divide_text_segments_by,
                text_segmentation_scale,
                self.align_language
            ], {
                "result": self.result,
                "align_language": self.align_language
            }):
                if self.align_language in ["ja", "zh", "zh-TW"]:
                    divide_text_segments_by += "|!|?|...|。"
                if text_segmentation_scale in ["word", "character"]:
                    self.result = linguistic_level_segments(
                        self.result,
                        text_segmentation_scale,
                    )
                elif divide_text_segments_by:
                    try:
                        self.result = break_aling_segments(
                            self.result,
                            break_characters=divide_text_segments_by,
                        )
                    except Exception as error:
                        logger.error(str(error))

            if not self.task_in_cache("diarize", [
                min_speakers,
                max_speakers,
                YOUR_HF_TOKEN[:len(YOUR_HF_TOKEN)//2],
                diarization_model
            ], {
                "result": self.result
            }):
                prog_disp("Diarizing...", 0.60, is_gui, progress=progress)
                diarize_model_select = diarization_models[diarization_model]
                self.result_diarize = diarize_speech(
                    base_audio_wav if not self.vocals else self.vocals,
                    self.result,
                    min_speakers,
                    max_speakers,
                    YOUR_HF_TOKEN,
                    diarize_model_select,
                )
                logger.debug("Diarize complete")
            self.result_source_lang = copy.deepcopy(self.result_diarize)

            if not self.task_in_cache("translate", [
                TRANSLATE_AUDIO_TO,
                translate_process
            ], {
                "result_diarize": self.result_diarize
            }):
                prog_disp("Translating...", 0.70, is_gui, progress=progress)
                lang_source = (
                    self.align_language
                    if self.align_language
                    else SOURCE_LANGUAGE
                )
                self.result_diarize["segments"] = translate_text(
                    self.result_diarize["segments"],
                    TRANSLATE_AUDIO_TO,
                    translate_process,
                    chunk_size=1800,
                    source=lang_source,
                )
                logger.debug("Translation complete")
                logger.debug(self.result_diarize)

        if get_translated_text:

            json_data = []
            for segment in self.result_diarize["segments"]:
                start = segment["start"]
                text = segment["text"]
                speaker = int(segment.get("speaker", "SPEAKER_00")[-2:]) + 1
                json_data.append(
                    {"start": start, "text": text, "speaker": speaker}
                )

            # Convert list of dictionaries to a JSON string with indentation
            json_string = json.dumps(json_data, indent=2)
            logger.info("Done")
            self.edit_subs_complete = True
            return json_string.encode().decode("unicode_escape")

        if get_video_from_text_json:

            if self.result_diarize is None:
                raise ValueError("Generate the transcription first.")
            # with open('text_json.json', 'r') as file:
            text_json_loaded = json.loads(text_json)
            for i, segment in enumerate(self.result_diarize["segments"]):
                segment["text"] = text_json_loaded[i]["text"]
                segment["speaker"] = "SPEAKER_{:02d}".format(
                    int(text_json_loaded[i]["speaker"]) - 1
                )

        # Write subtitle
        if not self.task_in_cache("subs_and_edit", [
            copy.deepcopy(self.result_diarize),
            output_format_subtitle,
            TRANSLATE_AUDIO_TO
        ], {
            "result_diarize": self.result_diarize
        }):
            if output_format_subtitle == "disable":
                self.sub_file = "sub_tra.srt"
            elif output_format_subtitle != "ass":
                self.sub_file = process_subtitles(
                    self.result_source_lang,
                    self.align_language,
                    self.result_diarize,
                    output_format_subtitle,
                    TRANSLATE_AUDIO_TO,
                )

            # Need task
            if output_format_subtitle != "srt":
                _ = process_subtitles(
                    self.result_source_lang,
                    self.align_language,
                    self.result_diarize,
                    "srt",
                    TRANSLATE_AUDIO_TO,
                )

            if output_format_subtitle == "ass":
                convert_ori = "ffmpeg -i sub_ori.srt sub_ori.ass -y"
                convert_tra = "ffmpeg -i sub_tra.srt sub_tra.ass -y"
                self.sub_file = "sub_tra.ass"
                run_command(convert_ori)
                run_command(convert_tra)

        format_sub = (
            output_format_subtitle
            if output_format_subtitle != "disable"
            else "srt"
        )

        if output_type == "subtitle":

            out_subs = []
            tra_subs = media_out(
                media_file,
                TRANSLATE_AUDIO_TO,
                video_output_name,
                format_sub,
                file_obj=self.sub_file,
            )
            out_subs.append(tra_subs)

            ori_subs = media_out(
                media_file,
                self.align_language,
                video_output_name,
                format_sub,
                file_obj=f"sub_ori.{format_sub}",
            )
            out_subs.append(ori_subs)
            logger.info(f"Done: {out_subs}")
            return out_subs

        if output_type == "subtitle [by speaker]":
            output = get_subtitle_speaker(
                media_file,
                result=self.result_diarize,
                language=TRANSLATE_AUDIO_TO,
                extension=format_sub,
                base_name=video_output_name,
            )
            logger.info(f"Done: {str(output)}")
            return output

        if "video [subtitled]" in output_type:
            output = media_out(
                media_file,
                TRANSLATE_AUDIO_TO + "_subtitled",
                video_output_name,
                "wav" if is_audio_file(media_file) else (
                    "mkv" if "mkv" in output_type else "mp4"
                ),
                file_obj=base_audio_wav if is_audio_file(media_file) else base_video_file,
                soft_subtitles=False if is_audio_file(media_file) else True,
                subtitle_files=output_format_subtitle,
            )
            msg_out = output[0] if isinstance(output, list) else output
            logger.info(f"Done: {msg_out}")
            return output

        if not self.task_in_cache("tts", [
            TRANSLATE_AUDIO_TO,
            tts_voice00,
            tts_voice01,
            tts_voice02,
            tts_voice03,
            tts_voice04,
            tts_voice05,
            tts_voice06,
            tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                tts_voice12,
                tts_voice13,
                tts_voice14,
                tts_voice15,
                tts_voice16,
                tts_voice17,
                tts_voice18,
                tts_voice19,
                VIDEO_OUTPUT_NAME,
                AUDIO_MIX,
                audio_accelerate,
                acceleration_rate_regulation_gui,
                volume_original_mix,
                volume_translated_mix,
                sub_type_output,
                edit_sub_check,  # TRUE BY DEFAULT
                dummy_false_check,  # dummy false
                subs_edit_space,
                avoid_overlap_gui,
                vocal_refinement_gui,
                literalize_numbers_gui,
                segment_duration_limit_gui,
                diarization_process_dropdown,
                translate_process_dropdown,
                input_srt,
                main_output_type,
                main_voiceless_track,
                voice_imitation_gui,
                voice_imitation_max_segments_gui,
                voice_imitation_vocals_dereverb_gui,
                voice_imitation_remove_previous_gui,
                voice_imitation_method_gui,
                wav_speaker_dereverb,
                text_segmentation_scale_gui,
                divide_text_segments_by_gui,
                soft_subtitles_to_video_gui,
                burn_subtitles_to_video_gui,
                enable_cache_gui,
                enable_custom_voice,
                workers_custom_voice,
                auto_clone_voices_gui,
                enable_lip_sync_gui,
                is_gui_dummy_check,
            ],
            outputs=subs_edit_space,
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
        )

        # Run translate tts and complete
        video_button.click(
            SoniTr.batch_multilingual_media_conversion,
            inputs=[
                video_input,
                blink_input,
                directory_input,
                HFKEY,
                PREVIEW,
                WHISPER_MODEL_SIZE,
                batch_size,
                compute_type,
                SOURCE_LANGUAGE,
                TRANSLATE_AUDIO_TO,
                min_speakers,
                max_speakers,
                tts_voice00,
                tts_voice01,
                tts_voice02,
                tts_voice03,
                tts_voice04,
                tts_voice05,
                tts_voice06,
                tts_voice07,
                tts_voice08,
                tts_voice09,
                tts_voice10,
                tts_voice11,
                tts_voice12,
                tts_voice13,
                tts_voice14,
                tts_voice15,
                tts_voice16,
                tts_voice17,
                tts_voice18,
                tts_voice19,
                VIDEO_OUTPUT_NAME,
                AUDIO_MIX,
                audio_accelerate,
                acceleration_rate_regulation_gui,
                volume_original_mix,
                volume_translated_mix,
                sub_type_output,
                edit_sub_check,
                dummy_false_check,
                subs_edit_space,
                avoid_overlap_gui,
                vocal_refinement_gui,
                literalize_numbers_gui,
                segment_duration_limit_gui,
                diarization_process_dropdown,
                translate_process_dropdown,
                input_srt,
                main_output_type,
                main_voiceless_track,
                voice_imitation_gui,
                voice_imitation_max_segments_gui,
                voice_imitation_vocals_dereverb_gui,
                voice_imitation_remove_previous_gui,
                voice_imitation_method_gui,
                wav_speaker_dereverb,
                text_segmentation_scale_gui,
                divide_text_segments_by_gui,
                soft_subtitles_to_video_gui,
                burn_subtitles_to_video_gui,
                enable_cache_gui,
                enable_custom_voice,
                workers_custom_voice,
                auto_clone_voices_gui,
                enable_lip_sync_gui,
                is_gui_dummy_check,
            ],
            outputs=video_output,
            trigger_mode="multiple",
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
        )

        # Run docs process
        docs_button.click(
            SoniTr.multilingual_docs_conversion,
            inputs=[
                text_docs,
                input_docs,
                directory_input_docs,
                docs_SOURCE_LANGUAGE,
                docs_TRANSLATE_TO,
                tts_documents,
                docs_OUTPUT_NAME,
                docs_translate_process_dropdown,
                docs_output_type,
                docs_chunk_size,
                enable_custom_voice,
                workers_custom_voice,
                start_page_gui,
                end_page_gui,
                videobook_width_gui,
                videobook_height_gui,
                videobook_bcolor_gui,
                docs_dummy_check,
            ],
            outputs=docs_output,
            trigger_mode="multiple",
        ).then(
            play_sound_alert, [play_sound_gui], [sound_alert_notification]
        )

    return app


def get_language_config(language_data, language=None, base_key="english"):
    base_lang = language_data.get(base_key)

    if language not in language_data:
        logger.error(
            f"Language {language} not found, defaulting to {base_key}"
        )
        return base_lang

    lg_conf = language_data.get(language, {})
    lg_conf.update((k, v) for k, v in base_lang.items() if k not in lg_conf)

    return lg_conf


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--theme",
        type=str,
        default="Taithrah/Minimal",
        help=(
            "Specify the theme; find themes in "
            "https://huggingface.co/spaces/gradio/theme-gallery;"
            " Example: --theme aliabid94/new-theme"
        ),
    )
    parser.add_argument(
        "--public_url",
        action="store_true",
        default=False,
        help="Enable public link",
    )
    parser.add_argument(
        "--logs_in_gui",
        action="store_true",
        default=False,
        help="Displays the operations performed in Logs",
    )
    parser.add_argument(
        "--verbosity_level",
        type=str,
        default="info",
        help=(
            "Set logger verbosity level: "
            "debug, info, warning, error, or critical"
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="english",
        help=" Select the language of the interface: english, spanish",
    )
    parser.add_argument(
        "--cpu_mode",
        action="store_true",
        default=False,
        help="Enable CPU mode to run the program without utilizing GPU acceleration.",
    )
    return parser


if __name__ == "__main__":

    parser = create_parser()

    args = parser.parse_args()
    # Simulating command-line arguments
    # args_list = "--theme aliabid94/new-theme --public_url".split()
    # args = parser.parse_args(args_list)

    set_logging_level(args.verbosity_level)

    for id_model in UVR_MODELS:
        download_manager(
            os.path.join(MDX_DOWNLOAD_LINK, id_model), mdxnet_models_dir
        )

    models_path, index_path = upload_model_list()

    SoniTr = SoniTranslate(cpu_mode=args.cpu_mode)

    lg_conf = get_language_config(language_data, language=args.language)

    app = create_gui(args.theme, logs_in_gui=args.logs_in_gui)

    app.queue()

    app.launch(
        max_threads=1,
        share=args.public_url,
        show_error=True,
        quiet=False,
        debug=(True if logger.isEnabledFor(logging.DEBUG) else False),
    )
