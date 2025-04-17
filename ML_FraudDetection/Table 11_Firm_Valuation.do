use "F:\Stata\Test\Firm_age_1950data.dta", clear

duplicates drop fyear gvkey, force

bysort gvkey (fyear): gen first_fyear = fyear if _n == 1
bysort gvkey (fyear): replace first_fyear = first_fyear[_n-1] if missing(first_fyear)
gen firm_age = fyear - first_fyear + 1
keep if fyear >= 2001 & fyear <= 2018

save "F:\Stata\Test\Firm_age_2001data.dta", replace


use "F:\Stata\Test\monthly_return.dta", clear

rename ret ret_monthly

gen datadate_num = date(date, "YMD")  
format datadate_num %td                  
gen mdate = mofd(datadate_num)             
format mdate %tm            

gen ret_num = real(ret_monthly)

gen lret = ln(1+ret_num)
rangestat (sum) lret (count) lret, interval(mdate -8 3) by(permno)
keep if lret_count==12
gen buy_hold_ret= exp(lret_sum) - 1


save "F:\Stata\Test\buy_hold_ret.dta", replace


use "F:\Stata\Test\2000-2018 COMPUSTAT ALL DATA.dta", clear

duplicates drop fyear gvkey, force

rename lpermno permno

gen datadate_num = date(datadate, "YMD")  
format datadate_num %td                  
gen mdate = mofd(datadate_num)             
format mdate %tm

merge 1:1 fyear gvkey using "F:\Stata\Test\prob.dta" 

drop _merge 

merge 1:1 fyear gvkey using "F:\Stata\Test\Firm_age_2001data.dta"

drop _merge 

duplicates drop mdate permno, force

merge 1:1 mdate permno using "F:\Stata\Test\buy_hold_ret.dta" 

*double-check duplicates
duplicates drop fyear gvkey, force
duplicates drop mdate permno, force

sort gvkey fyear
xtset gvkey fyear

*gen op_profit = oibdp / at
gen log_at = ln(at)
gen mve = csho * prcc_f
gen roa = ni / at
gen sales_growth = (sale - L.sale) / L.sale
gen leverage = (dltt + dlc) / at
gen capex_ratio = capx / at
gen tobin_q = (mve + at - ceq - txdb) / at
gen tobin_q_future1 = F.tobin_q

gen log_age = ln(firm_age)

*RD intensity
replace xrd=0 if xrd==.
gen rdi=xrd/at

*advertising intensity
replace xad=0 if xad==.
gen adi=xad/at

gen dividend_payout = (dvc>0|dvp>0)
gen mtb_ratio = mve/ceq

tostring sic, replace

replace sic="0100" if sic=="100"
replace sic="0200" if sic=="200"
replace sic="0700" if sic=="700"
replace sic="0800" if sic=="800"
replace sic="0900" if sic=="900"

gen first_two_digits = substr(sic, 1, 2)

egen miss_count = rowmiss(tobin_q_future1 prob_ada log_at capex_ratio roa leverage log_age)
drop if miss_count > 0
drop miss_count

keep if fyear >= 2001 & fyear <= 2018


winsor2 tobin_q_future1 prob_ada log_at capex_ratio roa leverage log_age rdi adi sales_growth buy_hold_ret dividend_payout mtb_ratio, cuts(1 99) replace by(fyear)

summarize tobin_q_future1 prob_ada log_at capex_ratio roa leverage log_age rdi adi sales_growth buy_hold_ret dividend_payout mtb_ratio, detail

*basic model1
areg tobin_q_future1 prob_ada log_at capex_ratio roa leverage log_age i.fyear, absorb(first_two_digits) cluster(gvkey)

outreg2 using ULTIMATE_MODELS_latest.xls, replace

*model2
areg tobin_q_future1 prob_ada log_at capex_ratio roa leverage log_age rdi adi sales_growth buy_hold_ret dividend_payout mtb_ratio i.fyear, absorb(first_two_digits) cluster(gvkey)

outreg2 using ULTIMATE_MODELS_latest.xls, append

save "F:\Stata\Test\ULTIMATE_DATASET_latest.dta", replace

