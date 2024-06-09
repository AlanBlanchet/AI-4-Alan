use pyo3::prelude::*;

#[pymodule]
pub fn rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn add(a: i32, h: i32) -> i32 {
    a + h
}
