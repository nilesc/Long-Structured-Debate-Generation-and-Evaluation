#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate pre-processed data with a trained model.
"""

import torch

# Add relevant preprocess packages
from collections import Counter
from itertools import zip_longest
import os
import shutil

from fairseq.data import indexed_dataset
from fairseq.tokenizer import Tokenizer
from multiprocessing import Pool

# Original generate packages
from fairseq import bleu, options, progress_bar, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer
from fairseq.utils import import_user_module

MAX_TURNS = 10

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Set up functions for multiturn
    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    def make_binary_dataset(input_prefix, output_prefix, lang, num_workers):
        dict = task.load_dictionary(dict_path(lang))
        print("| [{}] Dictionary: {} types".format(lang, len(dict) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Tokenizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        dict,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.IndexedDatasetBuilder(
            dataset_dest_file(args, output_prefix, lang, "bin")
        )
        merge_result(
            Tokenizer.binarize(
                input_file, dict, lambda t: ds.add_item(t), offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        print(
            "| [{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                dict.unk_word,
            )
        )

    def make_dataset(input_prefix, output_prefix, lang, num_workers=1):
        if args.output_format == "binary":
            make_binary_dataset(input_prefix, output_prefix, lang, num_workers)
        elif args.output_format == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)

    def make_all(lang):
        if args.multiturnpref:
            make_dataset(args.multiturnpref, "test", lang,
                    num_workers=args.workers)

    # Load dataset splits
    task = tasks.setup_task(args)

    # Multiturn tracking: prompt in test set, turn in debate
    turn = 0
    prompt = 1
    first_pass = True
    while first_pass or args.multiturn:
        if args.multiturn:
            # Set up first turn
            if turn == 0:
                multiturn_file = "{}{}".format(
                    args.multiturnpref, ("." + args.source_lang)
                )
                test_file = "{}{}".format(
                    args.testpref, ("." + args.source_lang)
                )
                with open(test_file, 'r', encoding='utf-8') as f:
                    for i in range(prompt):
                        line = f.readline()
                with open (multiturn_file, 'w', encoding='utf-8') as f:
                    f.write(line) 
                prompt += 1

            target = not args.only_source
            assert (
                args.multiturnpref
            ), "--multiturnpref must be set"
            if args.joined_dictionary:
                assert (
                        not args.srcdict or not args.tgtdict
                ), "cannot use both --srcdict and --tgtdict with --joined-dictionary"

                if args.srcdict:
                    src_dict = task.load_dictionary(args.srcdict)
                elif args.tgtdict:
                    src_dict = task.load_dictionary(args.tgtdict)
                else:
                    assert (
                        args.trainpref
                    ), "--trainpref must be set if --srcdict is not specified"
                    src_dict = build_dictionary({train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True)
                tgt_dict = src_dict
            else:
                if args.srcdict:
                    src_dict = task.load_dictionary(args.srcdict)
                else:
                    assert (
                        args.trainpref
                    ), "--trainpref must be set if --srcdict is not specified"
                    src_dict = build_dictionary([train_path(args.source_lang)], src=True)
            if target:
                if args.tgtdict:
                    tgt_dict = task.load_dictionary(args.tgtdict)
                else:
                    assert (
                        args.trainpref
                    ), "--trainpref must be set if --tgtdict is not specified"
                    tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)
            else:
                tgt_dict = None

            src_dict.save(dict_path(args.source_lang))
            if target and tgt_dict is not None:
                tgt_dict.save(dict_path(args.target_lang))
            
            make_all(args.source_lang)
            if target:
                make_all(args.target_lang)
            if first_pass:
                print("| Wrote preprocessed data to {}".format(args.destdir))
                print('| Generating multiturn debate')
            task.load_dataset('test')
        else:
            task.load_dataset(args.gen_subset)
            print('| {} {} {} examples'.format(args.data, args.gen_subset, len(task.dataset(args.gen_subset))))

        if first_pass:
            # Set dictionaries
            src_dict = task.source_dictionary
            tgt_dict = task.target_dictionary

            # Load ensemble
            print('| loading model(s) from {}'.format(args.path))
            models, _model_args = utils.load_ensemble_for_inference(
                args.path.split(':'), task, model_arg_overrides=eval(args.model_overrides),
            )

            # Optimize ensemble for generation
            for model in models:
                model.make_generation_fast_(
                    beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
                    need_attn=args.print_alignment,
                )
                if args.fp16:
                    model.half()

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        align_dict = utils.load_align_dict(args.replace_unk)

        # Load dataset (possibly sharded)
        itr = task.get_batch_iterator(
            dataset=task.dataset(args.gen_subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[model.max_positions() for model in models]
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=8,
            num_shards=args.num_shards,
            shard_id=args.shard_id,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)

        # Initialize generator
        gen_timer = StopwatchMeter()
        if args.score_reference:
            translator = SequenceScorer(models, task.target_dictionary)
        else:
            translator = SequenceGenerator(
                models, task.target_dictionary, beam_size=args.beam, minlen=args.min_len,
                stop_early=(not args.no_early_stop), normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen, unk_penalty=args.unkpen,
                sampling=args.sampling, sampling_topk=args.sampling_topk, sampling_temperature=args.sampling_temperature,
                diverse_beam_groups=args.diverse_beam_groups, diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len, no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

        if use_cuda:
            translator.cuda()

        # Generate and compute BLEU score
        if args.sacrebleu:
            scorer = bleu.SacrebleuScorer()
        else:
            scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
        num_sentences = 0
        has_target = True
        with progress_bar.build_progress_bar(args, itr) as t:
            if args.score_reference:
                translations = translator.score_batched_itr(t, cuda=use_cuda, timer=gen_timer)
            else:
                translations = translator.generate_batched_itr(
                    t, maxlen_a=args.max_len_a, maxlen_b=args.max_len_b,
                    cuda=use_cuda, timer=gen_timer, prefix_size=args.prefix_size,
                )

            wps_meter = TimeMeter()
            for sample_id, src_tokens, target_tokens, hypos in translations:

                # Process input and ground truth
                has_target = target_tokens is not None
                target_tokens = target_tokens.int().cpu() if has_target else None

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                for i, hypo in enumerate(hypos[:min(len(hypos), args.nbest)]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

                    if args.multiturn:
                        multiturn_file = "{}{}".format(
                            args.multiturnpref, ("." + args.source_lang)
                        )
                        output_file = "{}{}".format(
                            args.outputpref, ("." + args.target_lang)
                        )
                        with open(multiturn_file, 'r', encoding='utf-8') as f:
                            line = f.readline()
                        if turn < MAX_TURNS - 1:
                            with open(multiturn_file, 'w', encoding='utf-8') as f:
                                f.write(f'{line[:-1]} <EOA> {hypo_str}')
                            turn += 1 
                        elif turn == MAX_TURNS - 1:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(f'{line[:-1]} <EOA> {hypo_str}\n')
                            turn = 0

                    # Score only the top hypothesis
                    if has_target and i == 0:
                        if align_dict is not None or args.remove_bpe is not None:
                            # Convert back to tokens for evaluation with unk replacement and/or without BPE
                            target_tokens = tokenizer.Tokenizer.tokenize(
                                target_str, tgt_dict, add_if_not_exist=True)
                        if hasattr(scorer, 'add_string'):
                            scorer.add_string(target_str, hypo_str)
                        else:
                            scorer.add(target_tokens, hypo_tokens)

                wps_meter.update(src_tokens.size(0))
                t.log({'wps': round(wps_meter.avg)})
                num_sentences += 1

        print('| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
            num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
        if has_target:
            print('| Generate {} with beam={}: {}'.format(args.gen_subset, args.beam, scorer.result_string()))

        first_pass = False

def binarize(args, filename, dict, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.IndexedDatasetBuilder(
        dataset_dest_file(args, output_prefix, lang, "bin")
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Tokenizer.binarize(
        filename,
        dict,
        consumer,
        offset=offset,
        end=end,
        append_eos=append_eos
    )
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res

def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    lang_part = (
        ".{}-{}.{}".format(args.source_lang, args.target_lang, lang) if lang is not None else ""
    )
    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)

def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == '__main__':
    cli_main()
