from pathlib import PurePath, Path
import sys
import shutil
import logzero
from logzero import logger
import pandas as pd


# get project dir
pdir = PurePath("/YOUR/DIRECTORY/iex_intraday_equity_downloader")
data_dir = pdir / "data"
script_dir = pdir / "src" / "data"
sys.path.append(script_dir.as_posix())


pd.options.display.float_format = "{:,.4f}".format


# =============================================================================
# get current timestamp

now = pd.to_datetime("today")

# =============================================================================
# setup logger

log_dir = pdir / "logs" / "equity" / f"iex_downloader_log_{now.date()}.log"
logfile = PurePath(log_dir).as_posix()
log_format = (
    "%(color)s[%(levelname)1.1s %(asctime)s.%(msecs)03d"
    "%(module)s:%(lineno)d]%(end_color)s %(message)s"
)
formatter = logzero.LogFormatter(fmt=log_format, datefmt="%Y-%m-%d %I:%M:%S")
logzero.setup_default_logger(logfile=logfile, formatter=formatter)

# =============================================================================
# read intraday data into one dataframe

logger.info("reading all intraday data for today as dataframe...")
infp = PurePath(data_dir / "interim" / "intraday_store").as_posix()

try:
    df = pd.read_parquet(infp).drop_duplicates().dropna()
    if df.empty:
        logger.warn("empty dataframe for eod processing")
    # ==========================================================================
    # store intraday data into one compressed dataframe

    logger.info(
        "storing all intraday data for today as compressed parquet file..."
    )
    pq_dir = data_dir / "processed" / "intraday" / f"etf_{now.date()}.parq"
    outfp = PurePath(pq_dir)
    df.to_parquet(outfp, engine="fastparquet")

    # ==========================================================================
    # delete interim store

    logger.info("deleting all interim intraday data.")
    rmfp = Path(data_dir / "interim" / "intraday_store" / f"year={now.year}")
    shutil.rmtree(rmfp)

except Exception as e:
    logger.error(f"{e}\tlikely no data today: {now.date()}")
