package com.acme.service;

import com.acme.model.Contract;
import com.acme.model.Status;
import com.acme.repo.CustomerRepository;

public class ContractValidator {
    private final CustomerRepository repository;

    public ContractValidator(CustomerRepository repository) {
        this.repository = repository;
    }

    public boolean validateContract(Contract contract) {
        Status status = contract.status();
        return repository.exists(status);
    }
}
