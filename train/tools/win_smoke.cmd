@echo off
rem Pixie v3 Windows CPU smoke (no GPU: CUDA hidden). Run at low priority while v2 trains.
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set PYTHONUNBUFFERED=1
if "%PIXIE_ROOT%"=="" (
  echo set PIXIE_ROOT
  exit /b 1
)
set V3=%PIXIE_ROOT%\v3
set PY=%PIXIE_ROOT%\venv\Scripts\python.exe
set PAL=%PIXIE_ROOT%\data\subset150k\palette.npy
set V2TAB=%PIXIE_ROOT%\v2\clip_tok\clip_l14_token_embedding_f16.npy
set RUNS=%PIXIE_ROOT%\runs
set S=%V3%\smoke
if not exist %S% mkdir %S%
cd /d %V3%\code
echo === unit tests
%PY% test_v3.py
echo === rescore v1 A/B + v2 pilot (CPU, 64 rows)
%PY% rescore_v3.py --device cpu --limit 64 --data %V3%\data --cache %V3%\cache --pixelgen-root %V3%\code\pixelgen_copy --v2-clip-table %V2TAB% --palette %PAL% --v1 v1A=%RUNS%\full1p4m-L10-cont-cooldown\best.pt --v1 v1B=%RUNS%\full1p4m-L10-final32-cfg\best.pt --v2 v2pilot=%RUNS%\v2-L12-pilot\last.pt --out %S%\rescore_smoke.json
echo === train p50 tiny (with samples)
%PY% train_v3.py --arch p50 --clip %V3%\clip\clip_l14_text_l11_f32.pt --data %V3%\data --cache %V3%\cache --palette %PAL% --out %S%\p50 --device cpu --batch 8 --limit-rows 4000 --images 320 --warmup 5 --log-every 10 --eval-every-images 160 --save-every-images 160 --eval-limit 32 --width 64 --layers 2 --heads 4 --sample-per 1
echo === train pix tiny
%PY% train_v3.py --arch pix --clip %V3%\clip\clip_l14_text_l11_f32.pt --data %V3%\data --cache %V3%\cache --palette %PAL% --out %S%\pix --device cpu --batch 8 --limit-rows 4000 --images 160 --warmup 5 --log-every 10 --eval-every-images 160 --save-every-images 160 --eval-limit 32 --width 64 --layers 2 --heads 4 --no-samples
echo === train v2ft from the finished v2 pilot (tiny)
%PY% train_v3.py --arch v2ft --init %RUNS%\v2-L12-pilot\last.pt --v2-clip-table %V2TAB% --data %V3%\data --cache %V3%\cache --palette %PAL% --out %S%\v2ft --device cpu --batch 8 --limit-rows 4000 --images 160 --lr 5e-5 --warmup 5 --log-every 10 --eval-every-images 160 --save-every-images 160 --eval-limit 32 --no-samples
echo === compare
%PY% compare_v3.py %S% %S%
echo SMOKE_DONE
